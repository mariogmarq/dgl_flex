import copy

import torch
from flex.data import (Dataset, FedDataDistribution, FedDataset,
                       FedDatasetConfig)
from flex.model import FlexModel
from flex.pool import FlexPool, fed_avg
from flex.pool.decorators import (deploy_server_model,
                                  init_server_model, set_aggregated_weights, collect_clients_weights)
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


from torchvision import transforms
from torchvision.datasets import MNIST

EPOCHS = 1 # With 1 epoch, FedAvg is equivalent to FedSGD

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset():
    train_data = MNIST(root=".", train=True, download=True, transform=None)
    test_data = MNIST(root=".", train=False, download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)
    assert isinstance(flex_dataset, Dataset)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 2
    config.labels_per_node = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
    ]

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    assert isinstance(flex_dataset, FedDataset)
    flex_dataset["server"] = test_data

    return flex_dataset

# Make images between -1 and 1, since the generator outputs images between -1 and 1 due to tanh
mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: 2 * x - 1)]
)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 11) # 11 clases (10 dígitos + clase falsa)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 1024)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, img_size=28):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()
    server_flex_model["model"] = CNNModel()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    # Optimizer according to paper
    server_flex_model["optimizer_func"] = torch.optim.SGD
    server_flex_model["optimizer_kwargs"] = {
        "lr": 1e-3,
        "weight_decay": 1e-7,
        "momentum": 0
    }
    return server_flex_model

@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    new_flex_model = FlexModel()
    new_flex_model["model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["server_model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["discriminator"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    new_flex_model["optimizer_func"] = copy.deepcopy(server_flex_model["optimizer_func"])
    new_flex_model["optimizer_kwargs"] = copy.deepcopy(server_flex_model["optimizer_kwargs"])
    return new_flex_model

def train(client_flex_model: FlexModel, client_data: Dataset):
    model = client_flex_model["model"]
    criterion = client_flex_model["criterion"]
    model.train()
    model = model.to(device)
    torch_dataset = client_data.to_torchvision_dataset(transform=mnist_transforms)
    print(f"Client data: {len(torch_dataset)}")
    optimizer = client_flex_model["optimizer_func"](model.parameters(), **client_flex_model["optimizer_kwargs"])
    dataloader = DataLoader(
        torch_dataset, batch_size=64, shuffle=True, pin_memory=False
    )

    for _ in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss

@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    dev = aggregated_weights[0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    with torch.no_grad():
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(dev) + new)


@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return [(weight_dict[name] - server_dict[name].to(dev)).type(torch.float) for name in weight_dict]

def train_benign(pool: FlexPool):
    benign_client = pool.select(lambda id, role: id == 0)
    pool.servers.map(copy_server_model_to_clients, benign_client)
    losses = benign_client.map(train)
    print(f"Benign client loss: {losses}", flush=True)

# Work around for being able to concat with a TensorDataset
# since labels must be the same type
class TensorLabelDataset(torch.utils.data.Dataset):
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, idx):
        data, label = self.wrapped_dataset[idx]
        label = torch.tensor(label, dtype=torch.long)
        return data, label

def merge_dataset(client_data: Dataset, fake_images: torch.Tensor, fake_labels: torch.Tensor):
    dataset = TensorLabelDataset(client_data.to_torchvision_dataset(transform=mnist_transforms))
    # Insert the fake images into the dataset
    fake_images = fake_images.detach().cpu()
    fake_dataset = torch.utils.data.TensorDataset(fake_images, fake_labels)
    train_dataset = torch.utils.data.ConcatDataset([dataset, fake_dataset])
    return train_dataset

def optimize_gan(flex_model: FlexModel, client_data: Dataset, label: int = 0):
    if "generator" not in flex_model:
        flex_model["generator"] = Generator()
    discriminator = flex_model["discriminator"].to(device)
    generator = flex_model["generator"].to(device)
    criterion = flex_model["criterion"]

    # According to the paper use SGD, I found some stability problems with Adam
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.02, weight_decay=1e-5)

    generator.train()
    discriminator.train()
    generator_loss = torch.tensor(float("inf"))

    # Value of my choice
    GENERATOR_EPOCHS = 100

    for _ in range(GENERATOR_EPOCHS):
        generator_optimizer.zero_grad()
        noise = torch.randn(128, 100, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        fake_labels = torch.full((128,), label, dtype=torch.long, device=device)
        generator_loss = criterion(outputs, fake_labels)
        generator_loss.backward()
        generator_optimizer.step()


    return generator_loss

def train_with_fake_images(client_flex_model: FlexModel, client_data: Dataset):
    size = len(client_data) // 4 # Value of my choice
    fake_images = client_flex_model["generator"].to(device)(torch.randn(size, 100, device=device))
    fake_labels = torch.full((size,), 10, dtype=torch.long, device="cpu")
    train_dataset = merge_dataset(client_data, fake_images, fake_labels)

    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
    model = client_flex_model["model"]
    criterion = client_flex_model["criterion"]
    model.train()
    model = model.to(device)
    optimizer = client_flex_model["optimizer_func"](model.parameters(), **client_flex_model["optimizer_kwargs"])
    for _ in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    print(f"Malignant client loss: {running_loss}", flush=True)


def train_malicious(pool: FlexPool):
    malicious_client = pool.select(lambda id, role: id == 1)
    pool.servers.map(copy_server_model_to_clients, malicious_client)
    gen_loss = malicious_client.map(optimize_gan, label=0)[0]
    print(f"Generator loss: {gen_loss}", flush=True)
    malicious_client.map(train_with_fake_images)

def extract_fake_image(pool: FlexPool, i:int):
    malicious_client = pool.select(lambda id, role: id == 1)
    noise = torch.randn(1, 100, device=device)
    fake_image = malicious_client.map(lambda flex_model, _: flex_model["generator"].to(device)(noise))[0]
    fake_image = fake_image[0].detach().cpu().numpy()
    plt.imsave(f"images/fake_image_{i}.png", fake_image.squeeze(), cmap="gray")
    print("Fake image saved", flush=True)


def run_attack(pool: FlexPool):
    malicious_client = pool.select(lambda id, role: id == 1)
    benign_client = pool.select(lambda id, role: id == 0)
    for i in range(300):
        print(f"Round {i}")
        train_benign(pool)
        pool.servers.map(get_clients_weights, benign_client)
        pool.servers.map(fed_avg)
        pool.servers.map(set_agreggated_weights_to_server, pool.servers)
        train_malicious(pool)
        pool.servers.map(get_clients_weights, malicious_client)
        pool.servers.map(fed_avg)
        pool.servers.map(set_agreggated_weights_to_server, pool.servers)
        extract_fake_image(pool, i)




if __name__ == "__main__":
    flex_dataset = get_dataset()
    pool = FlexPool.client_server_pool(fed_dataset=flex_dataset, init_func=build_server_model)
    run_attack(pool)
