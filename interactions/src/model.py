import os
import sys
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

    def train_loop(self) -> None:
        '''Runs a single training epoch'''
        size = self._train_dataloader.train_size
        for batch, (x1, x2, y) in enumerate(self._train_dataloader):
            pred = self.forward(x1, x2)
            loss = self.loss(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print progress
            if batch % 100 == 0:
                loss, current = loss.item(), batch * size
                print(f'loss: {loss:5f} [{current:>5d}/{size:>5d}]')
            
    def train(self, train_dataset: CustomDataset, test_dataset: CustomDataset, epochs: int=10) -> None:
        '''Trains the model'''
        self._train_dataloader = train_dataset
        self._test_dataloader = test_dataset
        for epoch in range(epochs):
            self.train_loop()
            self.test()

    def test(self, dataloader: CustomDataset) -> None:
        '''Tests the model'''
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x1, x2, y in dataloader:
                pred = self.forward(x1, x2)
                test_loss += self.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def create_model(layers) -> MLP:
    '''Creates a new model'''
    model = MLP(layers)
    # if torch.cuda.is_available():
    #     model.cuda()
    return model
