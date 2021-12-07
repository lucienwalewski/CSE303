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
        # self.init_weights()

    def forward(self, x1: torch.tensor, x2: torch.tensor, normalize: bool = False) -> torch.tensor:
        '''Forwards pass '''
        if normalize:
            x1 = nn.functional.normalize(x1, dim=1)
            x2 = nn.functional.normalize(x2)
        x = torch.cat((x1, x2), dim=1)
        # features_norm = np.array(
        #     x) / np.sqrt(np.sum(np.array(x) ** 2, -1, keepdims=True))
        for layer in self.layers:
            x = torch.tanh(layer(x))
        x = torch.sigmoid(x)
        return x

    def init_weights(self) -> None:
        '''Initializes the weights of the model '''
        for layer in self.layers:
            nn.init.xavier_uniform_(
                layer.weight, gain=nn.init.calculate_gain('tanh'))

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

        # print(x1)
        # print(pred)
        pred = (pred > 0.5).float()
        correct += (pred == y).type(torch.float).sum().item()

        # Print progress
        if batch % 10 == 0 and batch > 0:
            accuracy = (100*correct)/(10 * train_dataloader.batch_size)
            correct = 0
            loss, current = loss.item(), batch * train_dataloader.batch_size
            print(
                f'loss: {loss:.3f}, accuracy: {accuracy:.2f}% [{current:>5d}/{size:>5d}]')

    return loss, accuracy


def train(model, train_dataset: DataLoader, test_dataset: DataLoader, loss_fn, optimizer, writer, epochs: int = 30) -> None:
    '''Trains the model'''
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        loss, accuracy = train_loop(model, train_dataset, loss_fn, optimizer)
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/accuracy', accuracy, epoch)
        loss, accuracy, precision, recall, f1 = test(model, test_dataset, loss_fn)
        writer.add_scalar('test/loss', loss, epoch)
        writer.add_scalar('test/accuracy', accuracy, epoch)
        writer.add_scalar('test/precision', precision, epoch)
        writer.add_scalar('test/recall', recall, epoch)
        writer.add_scalar('test/f1', f1, epoch)



def test(model, dataloader: DataLoader, loss_fn) -> Tuple:
    '''Tests the model'''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    test_loss, correct = 0, 0
    positives = 0

    with torch.no_grad():
        for (x1, x2), y in dataloader:
            y = y.to(torch.float)
            y = y.reshape(-1, 1)
            pred = model(x1, x2)
            pred = pred.to(torch.float)
            test_loss += loss_fn(pred, y).item()
            pred = (pred > 0.5).float()

            true_positives += ((pred == 1) & (y == 1)).type(torch.float).sum().item()
            true_negatives += ((pred == 0) & (y == 0)).type(torch.float).sum().item()

            false_positives += ((pred == 1) & (y == 0)).type(torch.float).sum().item()
            false_negatives += ((pred == 0) & (y == 1)).type(torch.float).sum().item()


    test_loss /= num_batches
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = 100 * ((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives))
    print(
        f'Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}')
    print(f'Percentage of negatives: {(true_negatives + false_positives) / (true_positives + true_negatives + false_positives + false_negatives)*100:>1f}%\n')
    return test_loss, accuracy, precision, recall, f1


def create_model(layers) -> MLP:
    '''Creates a new model'''
    model = MLP(layers)
    return model
