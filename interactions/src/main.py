import os
import sys
import numpy as np
import torch

from model import create_model
from load_data import create_dataset

if __name__ == '__main__':

    # Load the dataset
    tracks_path = os.path.join(os.path.dirname(__file__), '../data/tracks.json')
    labels_path = os.path.join(os.path.dirname(__file__), '../data/labels.json')
    data = create_dataset(tracks_path, labels_path)


    layers = [1024, 256, 1]
    model = create_model(layers)
    tensor1 = torch.rand(1, 512)
    tensor2 = torch.rand(1, 512)
    print(model.forward(tensor1, tensor2))