import os
import sys
import json
import random
import numpy as np
from typing import Dict, List, Tuple
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    Loads the data from the data directory.

    - Instances: List of tuples (x1, x2, y)
    where with x1, x2 the two input tensors and 
    y the ground truth label.
    """

    def __init__(self, instances: Dict, negative_ratio: int = None) -> None:
        super().__init__()
        self.positive_tracks = [(x1, x2) for x1, x2, y in instances['positive']]
        self.positive_labels = [y for x1, x2, y in instances['positive']]
        self.negative_tracks = [(x1, x2) for x1, x2, y in instances['negative']]
        self.negative_labels = [y for x1, x2, y in instances['negative']]
        self._positive_index = -1
        self._negative_index = -1
        self._negative_ratio = negative_ratio

    def __getitem__(self, index):
        return self.tracks[index], self.labels[index]

    def __iter__(self):
        return self

    def __next__(self):
        if self._negative_index >= len(self):
            self._negative_index = -1
            self._positive_index = -1
            raise StopIteration
        else:
            if (self._negative_index + 1) % self._negative_ratio == 0:
                self._negative_index += 1
                self._positive_index += 1
                return self.positive_tracks[self._positive_index], self.positive_labels[self._positive_index]
            self._negative_index += 1
            return self.negative_tracks[self._negative_index], self.negative_labels[self._negative_index]

    def __len__(self):
        return len(self.positive_tracks) + len(self.negative_tracks)


def load_tracks(tracks_path):
    """
    Loads the tracks from the given path.
    """
    return {int(k): v for k, v in json.load(open(tracks_path, 'r')).items()}


def load_labels(labels_path):
    """
    Loads the labels from the given path.
    """
    return json.load(open(labels_path, 'r'))


def create_datasets(tracks_path, labels_path, train_ratio: float = 0.8, negative_ratio: int = 0.75) -> Tuple[CustomDataset, CustomDataset]:
    """
    Creates a train and test dataset from the given paths.
    """
    tracks = load_tracks(tracks_path)
    labels = load_labels(labels_path)
    interaction_pairs = labels['interactions']
    no_interaction_pairs = labels['no interactions']

    positive_instances = []
    negative_instances = []
    # FIXME:
    # Create interaction pairs

    random.shuffle(positive_instances)
    random.shuffle(negative_instances)

    positive_split = len(positive_instances) // train_ratio
    negative_split = len(negative_instances) // train_ratio

    train_instances = {
        'positive': positive_instances[:positive_split], 'negative': negative_instances[:negative_split]}
    test_instances = {
        'positive': positive_instances[positive_split:], 'negative': negative_instances[negative_split:]}

    return CustomDataset(train_instances, negative_ratio), CustomDataset(test_instances)


def create_dataloaders(tracks_path, labels_path, batch_size=64, shuffle=True, num_workers=1, ratio=3) -> Tuple[DataLoader, DataLoader]:
    """
    Creates dataloaders from the given paths.
    """
    train_dataset, test_dataset = create_datasets(tracks_path, labels_path, ratio)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_dataloader, test_dataloader
