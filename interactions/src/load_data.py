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
        # Might require .read()
        self.tracks = {int(k): v for k, v in json.load(open(tracks_path, 'r')).items()}
        self.labels = json.load(open(labels_path, 'r'))
        self.interaction_pairs = self.labels['interactions']
        self.no_interaction_pairs = self.labels['no interactions']


        # To be fixed when proper data is given
        for track1, track2 in self.interaction_pairs:
            try:
                track1_dict = self.tracks[track1]
                track2_dict = self.tracks[track2]
                print(len(track1_dict))
                break
            except Exception as e:
                print(e)
                continue
        # # print(self.tracks.keys())
        # print(self.tracks['404'])
        # print(self.labels.keys())

    def __getitem__(self, index):
        return self.tracks[index], self.labels[index]

    def __len__(self):
        return len(self.tracks)


def create_dataset(tracks_path, labels_path):
    """
    Creates a dataset from the given paths.
    """
    return Data(tracks_path, labels_path)
