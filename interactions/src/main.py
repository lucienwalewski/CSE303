import os
import sys
import numpy as np
import torch

from model import create_model
from load_data import create_dataloaders

if __name__ == '__main__':

    # Load the dataset
    tracks_path = os.path.join(os.path.dirname(__file__), '../data/tracks.json')
    labels_path = os.path.join(os.path.dirname(__file__), '../data/labels.json')
    train_dataloader, test_dataloader = create_dataloaders(tracks_path, labels_path)
    for i, (x1, x2, y) in enumerate(train_dataloader):
        print(i, x1.shape, y.shape)
        break

    # # Create the model
    # layers = [1024, 256, 1]
    # model = create_model(layers)

    # # Train the model
    # model.train(train_dataloader, test_dataloader)