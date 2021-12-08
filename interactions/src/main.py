import os
import sys
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter

from model import create_model, train
from load_data import create_dataloaders, CustomDataset

if __name__ == '__main__':

    # wandb.init(project="my-test-project", entity="lucienwa")
    # wandb.config = {
    #     "learning_rate": 0.001,
    #     "epochs": 100,
    #     "batch_size": 128
    # }

    pca = False
    train_model = True

    # Load the dataset
    if pca:
        dir = '../data/features_pca.json'
    else:
        dir = '../data/features.json'
    data_path = os.path.join(
        os.path.dirname(__file__), dir)
    train_dataloader, test_dataloader = create_dataloaders(
        data_path)
    print(f'Loaded data from {data_path} into memory')

    # Create the model
    if pca:
        layers = [200, 10, 1]
    else:
        layers = [1024, 256, 1]
    model = create_model(layers)
    model.train()

    writer = SummaryWriter()

    # Train the model
    loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train(model, train_dataloader, test_dataloader,
          loss, optimizer, writer, epochs=60)

    writer.close()

    # Save the model
    model.save('models/model.pt')
