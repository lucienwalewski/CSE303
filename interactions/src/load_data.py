import os
import sys
import json
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    Loads the data from the data directory.

    - Instances: List of tuples (x1, x2, y)
    where with x1, x2 the two input tensors and 
    y the ground truth label.
    """

    def __init__(self, instances: Dict, train_negative_ratio: int = 3) -> None:
        super().__init__()
        self.positive_tracks = [(x1, x2)
                                for x1, x2 in instances['positive']]
        self.negative_tracks = [(x1, x2)
                                for x1, x2 in instances['negative']]
        self._positive_index = -1
        self._negative_index = -1
        self._current_loop_index = 0
        self._negative_ratio = train_negative_ratio
        self._train_loop_count = 0
        # Number of negative training instances per train loop
        self._train_neg_number = train_negative_ratio * \
            len(self.positive_tracks)

    def __getitem__(self, index):
        return self.tracks[index], self.labels[index]

    def __iter__(self):
        return self

    def __next__(self):
        if self._train_loop_count < self.number_train_loops - 1:
            if self._current_loop_index >= self.train_loop_size:
                self._positive_index = -1
                self._current_loop_index = 0
                self._train_loop_count += 1
                raise StopIteration
            if (self._negative_index + self._positive_index + 1) % (self._negative_ratio + 1) == 0:
                self._positive_index += 1
                self._current_loop_index += 1
                return self.positive_tracks[self._positive_index], 1
            self._negative_index += 1
            self._current_loop_index += 1
            return self.negative_tracks[self._negative_index], 0
        if self._negative_index >= len(self.negative_tracks) - 1:
            self._positive_index = -1
            self._negative_index = -1
            self._current_loop_index = -1
            self._train_loop_count = 0
            raise StopIteration
        # print(self._positive_index, self._negative_index, self._current_loop_index)
        if (self._negative_index + self._positive_index + 1) % (self._negative_ratio + 1) == 0:
            self._positive_index += 1
            self._current_loop_index += 1
            return self.positive_tracks[self._positive_index], 1
        self._negative_index += 1
        self._current_loop_index += 1
        return self.negative_tracks[self._negative_index], 0

    def __len__(self):
        return len(self.positive_tracks) + len(self.negative_tracks)

    @property
    def number_positives(self):
        return len(self.positive_tracks)

    @property
    def number_negatives(self):
        return len(self.negative_tracks)

    @property
    def train_loop_size(self):
        '''Number of instances in a single train loop'''
        return len(self.positive_tracks) + self._train_neg_number

    @property
    def train_neg_size(self):
        '''Number of negative training instances per train loop'''
        return self._train_neg_number

    @property
    def entire_size(self):
        return len(self)

    @property
    def number_train_loops(self):
        return (len(self.negative_tracks) // self._train_neg_number) + 1


def load_tracks(tracks_path):
    """
    Loads the tracks from the given path.
    """
    return json.load(open(tracks_path, 'r'))


def load_labels(labels_path):
    """
    Loads the labels from the given path.
    """
    return json.load(open(labels_path, 'r'))


def create_positive_instances(tracks: List, labels: List) -> List:
    """
    Creates positive instances from the given tracks and labels.
    """
    positive_instances = []
    for i, interaction1 in enumerate(tracks):
        for j, interaction2 in enumerate(tracks[i+1:]):
            j += i + 1
            ti1, ti2 = interaction1['track1'], interaction1['track2']
            tj1, tj2 = interaction2['track1'], interaction2['track2']

            if ti1 == tj2 and ti2 == tj1 and interaction1['segment'] == interaction2['segment'] and ([ti1, ti2] in labels or [ti2, ti1] in labels):
                positive_instances.append(
                    (torch.tensor(interaction1['features']), torch.tensor(interaction2['features'])))
    return positive_instances


def create_type1_negatives(tracks: List, labels: List) -> List:
    """
    Creates type1 negatives from the given tracks and labels.
    """
    negative_instances = []
    for i, interaction1 in enumerate(tracks):
        for j, interaction2 in enumerate(tracks[i+1:]):
            j += i
            ti1, ti2 = interaction1['track1'], interaction1['track2']
            tj1, tj2 = interaction2['track1'], interaction2['track2']

            if ti1 == tj2 and ti2 == tj1 and interaction1['segment'] == interaction2['segment'] and ([ti1, ti2] not in labels and [ti2, ti1] not in labels):
                negative_instances.append(
                    (torch.tensor(interaction1['features']), torch.tensor(interaction2['features'])))
    return negative_instances


def create_type2_negatives(tracks: List, labels: List) -> List:
    """
    Creates type2 negatives from the given tracks and labels. These
    are pairs from the same tracks but on different segments.
    """
    negative_instances = []
    for i, interaction1 in enumerate(tracks):
        for j, interaction2 in enumerate(tracks[i+1:]):
            j += i
            ti1, ti2 = interaction1['track1'], interaction1['track2']
            tj1, tj2 = interaction2['track1'], interaction2['track2']
            if ti1 == tj2 and ti2 == tj1 and interaction1['segment'] != interaction2['segment']:
                negative_instances.append(
                    (torch.tensor(interaction1['features']), torch.tensor(interaction2['features'])))
    return negative_instances


def create_type3_negatives(tracks: List, labels: List) -> List:
    """
    Creates type3 negatives from the given tracks and labels. These
    are pairs from different tracks.
    """
    # FIXME
    pass


def create_negative_instances(tracks: List, labels: List) -> List:
    """
    Creates negative instances from the given tracks and labels.
    """
    negative_instances = []
    negative_instances += create_type1_negatives(tracks, labels)
    negative_instances += create_type2_negatives(tracks, labels)
    return negative_instances


def check_duplicate_tracks(tracks: List) -> bool:
    """
    Checks if the given tracks are duplicates.
    """
    seen = defaultdict(int)
    for track in tracks:
        seen[(track['track1'], track['track2'],
              track['segment'][0], track['segment'][1])] += 1
    return any((count > 1 for count in seen.values()))


def create_datasets(tracks_path, labels_path, train_ratio: float = 0.8, negative_ratio: int = 0.75) -> Tuple[CustomDataset, CustomDataset]:
    """
    Creates a train and test dataset from the given paths.
    """
    tracks = load_tracks(tracks_path)
    interaction_pairs = load_labels(labels_path)
    # FIXME:
    # try:
    #     assert check_duplicate_tracks(tracks) == False
    # except Exception as e:
    #     print(f'Malformed data: Duplicate definitions of certain tracks')
    #     sys.exit(1)
    positive_instances = create_positive_instances(tracks, interaction_pairs)
    negative_instances = create_negative_instances(tracks, interaction_pairs)

    random.shuffle(positive_instances)
    random.shuffle(negative_instances)

    positive_split = int(len(positive_instances) * train_ratio)
    negative_split = int(len(negative_instances) * train_ratio)

    train_instances = {
        'positive': positive_instances[:positive_split], 'negative': negative_instances[:negative_split]}
    test_instances = {
        'positive': positive_instances[positive_split:], 'negative': negative_instances[negative_split:]}

    return CustomDataset(train_instances, train_negative_ratio=3), CustomDataset(test_instances)


def create_dataloaders(tracks_path, labels_path, batch_size=64, shuffle=True, num_workers=1, ratio=3) -> Tuple[DataLoader, DataLoader]:
    """
    Creates dataloaders from the given paths.
    """
    train_dataset, test_dataset = create_datasets(
        tracks_path, labels_path)

    # return train_dataset, test_dataset
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_dataloader, test_dataloader
