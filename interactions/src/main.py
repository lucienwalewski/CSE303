import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter

from model import create_model, train
from load_data import create_dataloaders

if __name__ == '__main__':

    # Load the dataset
    tracks_path = os.path.join(
        os.path.dirname(__file__), '../data/tracks.json')
    labels_path = os.path.join(
        os.path.dirname(__file__), '../data/labels.json')
    train_dataloader, test_dataloader = create_dataloaders(
        tracks_path, labels_path)
    
    # Create the model
    layers = [1024, 256, 1]
    model = create_model(layers)

    writer = SummaryWriter('log')
    writer.add_graph(model, (torch.rand(1, 512), torch.rand(1, 512)))
    writer.close()

    # Train the model
    loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train(model, train_dataloader, test_dataloader, loss, optimizer)

    # Test random tensors
    # x1 = torch.randn(64, 512)
    # x2 = torch.randn(64, 512)
    # y = torch.randn(64, 1)
    # y = y.to(torch.float)
    # y = y.reshape(-1, 1)
    # pred = model(x1, x2)
    # print(pred)
    # print(loss(pred, y.reshape(-1, 1)))
    # for param in model.parameters():
    #     print(param.shape)
    #     print(param)

    # Save the model

