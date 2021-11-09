import os
import sys
import json
import torch.utils.data as data

class Data(data.Dataset):
    """
    Loads the data from the data directory.
    """
    def __init__(self, tracks_path, labels_path) -> None:
        super().__init__()
        # FIXME
        self.tracks = json.load(open(tracks_path, 'r')) # Might require .read()
        self.labels = json.load(open(labels_path, 'r')) 


    def __getitem__(self, index):
        return self.tracks[index], self.labels[index]

    def __len__(self):
        return len(self.tracks)


def create_dataset(tracks_path, labels_path):
    """
    Creates a dataset from the given paths.
    """
    return Data(tracks_path, labels_path)