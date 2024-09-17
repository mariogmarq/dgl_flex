import copy
import os

import torch
from torch.autograd import grad
from flex.data import (Dataset, FedDataDistribution, FedDataset,
                       FedDatasetConfig)
from flex.model import FlexModel
from flex.pool import FlexPool
from flex.pool.decorators import (deploy_server_model,
                                  init_server_model)
from matplotlib import pyplot as plt
from typing import List
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import FashionMNIST

CLIENTS_PER_ROUND = 10
EPOCHS = 5
N_ROUNDS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

DIR = f"{os.path.dirname(__file__)}/metrics"
writer = SummaryWriter(f"{DIR}/dgl")

figures = []

def get_dataset():
    train_data = FashionMNIST(root=".", train=True, download=True, transform=None)
    test_data = FashionMNIST(root=".", train=False, download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)
    assert isinstance(flex_dataset, Dataset)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200

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
    def __init__(self, num_classes=10):
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
    new_flex_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    new_flex_model["optimizer_func"] = copy.deepcopy(server_flex_model["optimizer_func"])
    new_flex_model["optimizer_kwargs"] = copy.deepcopy(server_flex_model["optimizer_kwargs"])
    return new_flex_model

def get_client_gradient(client_flex_model: FlexModel, client_data: Dataset):
    model = client_flex_model["model"]
    criterion = client_flex_model["criterion"]
    model.train()
    model = model.to(device)
    test_dataset = client_data.to_torchvision_dataset(transform=mnist_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, pin_memory=False
    )
    for data, target in test_dataloader:
        model.zero_grad()
        break
    
    data, target = data.to(device), target.to(device)
    data = data / data.max()
    preds = model(data)
    loss = criterion(preds, target)
    return data, target, [g.detach().clone() for g in grad(loss, model.parameters())]
    
def extract_label(grads: List[torch.Tensor]):
    last_layer_grad = grads[-1]
    # The label is the index of the value with different sign
    signs = torch.sign(last_layer_grad)
    different_sign_idx = torch.where(signs != signs[0])[0]
    assert different_sign_idx.numel() == 1
    
    onehot = torch.zeros_like(last_layer_grad, device=device)
    onehot[different_sign_idx] = 1
    # add batch dimension
    onehot = onehot.unsqueeze(0)
    return onehot


def optimize_dummy(flex_model: FlexModel, _: Dataset, grads: List[torch.Tensor], label: torch.Tensor):
    model = flex_model["model"]
    model.to(device)
    assert isinstance(model, nn.Module)
    dummy_x = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)

    optimizer = torch.optim.LBFGS([dummy_x])
    
    def closure():
        optimizer.zero_grad()
        pred = model(dummy_x)
        loss = flex_model["criterion"](pred, label)
        network_grads = grad(loss, model.parameters(), create_graph=True)
        
        distance = 0
        for g1, g2 in zip(network_grads, grads):
            distance += ((g1 - g2) ** 2).sum()
        distance.backward()
        return distance
        
    
    for step in range(1, 301):
        optimizer.step(closure)
        
        if step % 100 == 0 and step != 0:
            distance = closure()
            print(f"Step {step}: {distance}")
    
    plt.imsave("final.png", dummy_x.cpu().detach().numpy().reshape(28, 28), cmap="gray")
    
def iDGL(pool: FlexPool):
    client_to_leak = pool.clients.select(1)
    pool.servers.map(copy_server_model_to_clients, client_to_leak)
    data, target, grads = client_to_leak.map(get_client_gradient)[0]
    label = extract_label(grads)
    print(target, label)
    pool.servers.map(optimize_dummy, grads=grads, label=label)
    
    assert data is not None
    data = data[0]
    data = data.cpu().reshape(28, 28)
    
    plt.imsave("original.png", data.numpy(), cmap="gray")
    

if __name__ == "__main__":
    flex_dataset = get_dataset()
    pool = FlexPool.client_server_pool(fed_dataset=flex_dataset, init_func=build_server_model)
    iDGL(pool)
    writer.flush()
    writer.close()
