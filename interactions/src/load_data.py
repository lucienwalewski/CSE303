import pandas as pd
import os
import sys
import json
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split


class CustomDataset(Dataset):
    def __init__(self, instances, shuffle=False) -> None:
        super().__init__()
        self._instances = instances
        if shuffle:
            random.shuffle(self._instances)
        self._index = -1

    def __len__(self):
        return len(self._instances)

    def __getitem__(self, index):
        return self._instances[index]


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
    pass
    # positive_instances = []
    # for i, interaction1 in enumerate(tracks):
    #     for j, interaction2 in enumerate(tracks[i+1:]):
    #         j += i + 1
    #         ti1, ti2 = interaction1['track1'], interaction1['track2']
    #         tj1, tj2 = interaction2['track1'], interaction2['track2']

    #         if ti1 == tj2 and ti2 == tj1 and interaction1['segment'] == interaction2['segment'] and

    # positive_instances = []
    # sample = []
    # for i, interaction1 in enumerate(tracks):
    #     for j, interaction2 in enumerate(tracks[i+1:]):
    #         j += i + 1
    #         ti1, ti2 = interaction1['track1'], interaction1['track2']
    #         tj1, tj2 = interaction2['track1'], interaction2['track2']

    #         if ti1 == tj2 and ti2 == tj1 and interaction1['segment'] == interaction2['segment'] and ([ti1, ti2] in labels or [ti2, ti1] in labels):
    #             positive_instances.append(
    #                 (torch.tensor(interaction1['features']), torch.tensor(interaction2['features'])))
    #             if ti1 == 69 and ti2 == 70:
    #                 sample.append(
    #                     {'track1': ti1, 'track2': ti2, 'segment': interaction1['segment'], 'features1': interaction1['features'], 'features2': interaction2['features']})

    # with open('sample.json', 'w') as f:
    #     json.dump(sample, f)

    # return positive_instances


def create_type1_negatives(tracks: List, labels: List) -> List:
    """
    Creates type1 negatives from the given tracks and labels. These pairs
    are from the same tracks and same segments but are known to be negative.
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


def create_datasets(tracks_path, labels_path, train_ratio: float = 0.8, batch_size: int = 64, negative_frac: float = 0.5):
    """
    Creates a train and test dataset from the given paths.
    """
    # Load the tracks and labels
    tracks = load_tracks(tracks_path)
    interaction_pairs = load_labels(labels_path)

    # Create the positive and negative instances
    positive_instances = create_positive_instances(tracks, interaction_pairs)
    negative_instances = create_negative_instances(tracks, interaction_pairs)

    # Create the full dataset
    dataset = CustomDataset([(pos, 1) for pos in positive_instances] + [
        (neg, 0) for neg in negative_instances], shuffle=True)

    # Split the dataset into train and test
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

    # Create weighted random sampler for the train set
    train_classes = [label for _, label in train_set]
    class_count = Counter(train_classes)
    class_weights = torch.Tensor([len(train_classes) / count
                                 for count in pd.Series(class_count).sort_index().values])
    class_weights[0] *= ((1 / (1 - negative_frac)) - 1)

    sample_weights = [0] * len(train_set)
    for idx, (_, label) in enumerate(train_set):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    return train_loader, DataLoader(val_set, batch_size=batch_size, shuffle=False)


def create_datasets2(path, train_ratio: float = 0.75, batch_size: int = 64, negative_frac: float = 0.75):
    data = json.load(open(path))
    positive_instances = [(torch.tensor(ex['person1_features']), torch.tensor(ex['person2_features']))
                          for ex in data if ex['label'] == 1]
    negative_instances = [(torch.tensor(ex['person1_features']), torch.tensor(ex['person2_features']))
                          for ex in data if ex['label'] == 0]

    # Create the full dataset
    dataset = CustomDataset([(pos, 1) for pos in positive_instances] + [
        (neg, 0) for neg in negative_instances], shuffle=True)

    # Split the dataset into train and test
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

    val_pos = [ex for ex, label in val_set if label == 1]
    val_neg = [ex for ex, label in val_set if label == 0]
    val_set = CustomDataset([(pos, 1) for pos in val_pos] + [(neg, 0)
                            for neg in random.sample(val_neg, len(val_pos))])

    # Create weighted random sampler for the train set
    train_classes = [label for _, label in train_set]
    class_count = Counter(train_classes)
    class_weights = torch.Tensor([len(train_classes) / count
                                 for count in pd.Series(class_count).sort_index().values])
    class_weights[0] *= ((1 / (1 - negative_frac)) - 1)

    sample_weights = [0] * len(train_set)
    for idx, (_, label) in enumerate(train_set):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    return train_loader, DataLoader(val_set, batch_size=batch_size, shuffle=False)


def create_dataloaders(tracks_path, batch_size=64, shuffle=True, num_workers=1, ratio=3) -> Tuple[DataLoader, DataLoader]:
    """
    Creates dataloaders from the given paths.
    """
    train_dataset, test_dataset = create_datasets2(
        tracks_path)

    return train_dataset, test_dataset
