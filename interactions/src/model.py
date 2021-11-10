import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Tuple


class MLP(nn.Module):
    def __init__(self, layers: List[int]) -> None:
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1])
             for i in range(len(layers) - 1)]
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.BCELoss()

    def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
        '''Forwards pass '''
        x = torch.cat((x1, x2), dim=1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(x)
        return x

    def save(self, path: str) -> None:
        '''Saves the model to a file '''
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        '''Loads the model from a file '''
        self.load_state_dict(torch.load(path))

    def train(self, dataloader: DataLoader) -> None:
        '''Trains the model'''
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            print(batch)


def create_model(layers) -> MLP:
    '''Creates a new model'''
    model = MLP(layers)
    if torch.cuda.is_available():
        model = model.cuda()
    return model
