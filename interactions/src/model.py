import os
import sys
import numpy as np
from numpy import negative
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from typing import List, Tuple
from load_data import CustomDataset


class MLP(nn.Module):
    def __init__(self, layers: List[int]) -> None:
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1])
             for i in range(len(layers) - 1)]
        )
        self.init_weights()

    def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
        '''Forwards pass '''
        x = torch.cat((x1, x2), dim=1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(x)
        return x
    
    def init_weights(self) -> None:
        '''Initializes the weights of the model '''
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))


    def save(self, path: str) -> None:
        '''Saves the model to a file '''
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        '''Loads the model from a file '''
        self.load_state_dict(torch.load(path))

def train_loop(model, train_dataloader, loss_fn, optimizer) -> None:
    '''Runs a single training epoch'''
    size = len(train_dataloader.dataset)
    correct = 0
    for batch, ((x1, x2), y) in enumerate(train_dataloader):
        pred = model(x1, x2)
        y = y.to(torch.float)
        y = y.reshape(-1, 1)
        pred = pred.to(torch.float)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (pred >= 0.5).float()
        correct += (pred == y).type(torch.float).sum().item()

        # Print progress
        if batch % 10 == 0 and batch > 0:
            accuracy = (100*correct)/(10 * train_dataloader.batch_size)
            correct = 0
            loss, current = loss.item(), batch * train_dataloader.batch_size
            print(f'loss: {loss:.3f}, accuracy: {accuracy:.2f}% [{current:>5d}/{size:>5d}]')
        
def train(model, train_dataset: DataLoader, test_dataset: DataLoader, loss_fn, optimizer, epochs: int=10) -> None:
    '''Trains the model'''
    for epoch in range(epochs):
        train_loop(model, train_dataset, loss_fn, optimizer)
        test(model, test_dataset, loss_fn)

def test(model, dataloader: DataLoader, loss_fn) -> None:
    '''Tests the model'''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (x1, x2), y in dataloader:
            y = y.to(torch.float)
            y = y.reshape(-1, 1)
            pred = model(x1, x2)
            pred = pred.to(torch.float)
            test_loss += loss_fn(pred, y).item()
            pred = (pred >= 0.5).float()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def create_model(layers) -> MLP:
    '''Creates a new model'''
    model = MLP(layers)
    return model
