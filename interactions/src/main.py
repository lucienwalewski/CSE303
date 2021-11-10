import os
import sys
import numpy as np
import torch

from model import create_model
from load_data import create_dataloaders, create_dataset

if __name__ == '__main__':

    # Load the dataset
    tracks_path = os.path.join(os.path.dirname(__file__), '../data/tracks.json')
    labels_path = os.path.join(os.path.dirname(__file__), '../data/labels.json')
    train_dataloader, test_dataloader = create_dataloaders(tracks_path, labels_path)

    # Create the model
    layers = [1024, 256, 1]
    model = create_model(layers)

    # Train the model
    model.train(train_dataloader, test_dataloader)