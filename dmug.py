import copy
import os

import torch
from flex.data import (Dataset, FedDataDistribution, FedDataset,
                       FedDatasetConfig)
from flex.model import FlexModel
from flex.pool import FlexPool, fed_avg
from flex.pool.decorators import (deploy_server_model,
                                  init_server_model, set_aggregated_weights, collect_clients_weights)
from matplotlib import pyplot as plt
from typing import List
import numpy as np
from torch import nn
from torch.utils.data import DataLoader


from torchvision import transforms
from torchvision.datasets import FashionMNIST

EPOCHS = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

figures = []

def get_dataset():
    train_data = FashionMNIST(root=".", train=True, download=True, transform=None)
    test_data = FashionMNIST(root=".", train=False, download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)
    assert isinstance(flex_dataset, Dataset)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 2
    config.labels_per_node = [
        [0, 1],
        [2, 3]
    ]

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    data_threshold = 30
    # Get users with more than 30 items
    print("All users", len(flex_dataset))
    cids = list(flex_dataset.keys())
    for k in cids:
        if len(flex_dataset[k]) < data_threshold:
            del flex_dataset[k]

    print("Filtered users", len(flex_dataset))

    assert isinstance(flex_dataset, FedDataset)
    flex_dataset["server"] = test_data

    return flex_dataset

mnist_transforms = transforms.Compose(
    [transforms.ToTensor()]
)



class CNNModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(14*14*64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        return self.fc(x)


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
            nn.Sigmoid(),
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
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
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
        torch_dataset, batch_size=128, shuffle=True, pin_memory=False
    )

    for _ in range(EPOCHS + 6):
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
    print(f"Benign client loss: {losses}")

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
    dataloader = DataLoader(client_data.to_torchvision_dataset(transform=mnist_transforms), batch_size=128, shuffle=True, pin_memory=False)
    
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.train()
    discriminator.train()
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        outputs = discriminator(inputs)
        loss_real = criterion(outputs, labels)

        fake_images = generator(torch.randn(128, 100, device=device))
        fake_labels = torch.full((128,), label, dtype=torch.long, device=device)
        true_labels = torch.full((128,), 10, dtype=torch.long, device=device)
        fake_outputs = discriminator(fake_images)
        generator_loss = criterion(fake_outputs, fake_labels)
        dis_fake_loss = criterion(fake_outputs, true_labels)
        discriminator_loss = loss_real + dis_fake_loss
        discriminator_loss.backward(retain_graph=True)
        generator_loss.backward()
        
        discriminator_optimizer.step()
        generator_optimizer.step()
    
    return generator_loss, discriminator_loss

def train_with_fake_images(client_flex_model: FlexModel, client_data: Dataset):
    size = len(client_data)
    fake_images = client_flex_model["generator"].to(device)(torch.randn(size, 100, device=device))
    fake_labels = torch.full((size,), 4, dtype=torch.long, device="cpu")
    train_dataset = merge_dataset(client_data, fake_images, fake_labels)

    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=False)
    model = client_flex_model["model"]
    criterion = client_flex_model["criterion"]
    model.train()
    model = model.to(device)
    optimizer = client_flex_model["optimizer_func"](model.parameters(), **client_flex_model["optimizer_kwargs"])
    for _ in range(EPOCHS + 6):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    print(f"Malignant client loss: {running_loss}")
    

def train_malicious(pool: FlexPool):
    malicious_client = pool.select(lambda id, role: id == 1)
    pool.servers.map(copy_server_model_to_clients, malicious_client)
    for i in range(16):
        gen_loss, dis_loss = malicious_client.map(optimize_gan, label=0)[0]
        print(f"[EPOCH {i+1}/10] Generator loss: {gen_loss}, Discriminator loss: {dis_loss}")
    malicious_client.map(train_with_fake_images)
    
def extract_fake_image(pool: FlexPool, i:int):
    malicious_client = pool.select(lambda id, role: id == 1)
    noise = torch.randn(64, 100, device=device)
    fake_image = malicious_client.map(lambda flex_model, _: flex_model["generator"].to(device)(noise))[0]
    fake_image = fake_image[0].detach().cpu().numpy()
    # make sure image is between 0 and 1, if not, normalize
    plt.imsave(f"fake_image_{i}.png", fake_image.squeeze(), cmap="gray")
    print("Fake image saved")
    

def run_attack(pool: FlexPool):
    # Warmup
    train_benign(pool)
    benign_client = pool.select(lambda id, role: id == 0)
    pool.servers.map(get_clients_weights, benign_client)
    pool.servers.map(fed_avg)
    pool.servers.map(set_agreggated_weights_to_server, pool.servers)
    for i in range(10):
        print(f"Round {i}")
        train_benign(pool)
        train_malicious(pool)
        pool.servers.map(get_clients_weights, pool.clients)
        pool.servers.map(fed_avg)
        pool.servers.map(set_agreggated_weights_to_server, pool.servers)
        extract_fake_image(pool, i)

