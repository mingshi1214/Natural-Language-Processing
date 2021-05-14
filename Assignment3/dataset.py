from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Tuple, List
import random


Tensor = torch.Tensor


def zscore_normalize(data: Tensor) -> Tensor:
    """
    Perform Z-Score Normalization on the given data. It is assumed that the data given has shape (batch, seq, features).
    """
    mean = data.mean(dim=2).unsqueeze(2)
    std = data.std(dim=2).unsqueeze(2)
    normed_data = (data - mean) / std
    # replace the NaN's with 0's -- NaN's will occur at the padded positions
    normed_data[normed_data != normed_data] = 0

    return normed_data


def minmax_normalize(data: Tensor) -> Tensor:
    """
    Perform Min-Max normalization to scale the data into the [0, 1] range.
    """
    mins = data.min(dim=2)[0]
    maxs = data.max(dim=2)[0]

    scaled = (data - mins.unsqueeze(2)) / (maxs - mins).unsqueeze(2)
    scaled[torch.isnan(scaled)] = 0

    return scaled


class MFCCDataset(Dataset):
    """
    A PyTorch dataset for MFCC data.
    """
    def __init__(self, inputs_paths: List[str], labels: List[int], shuffle: bool=True) -> None:
        """
        Initializes an instance of the MFCC dataset.

        <source> should be the path to the directory containing subdirectories of data.
        """
        super(MFCCDataset, self).__init__()
        self.paths = inputs_paths
        self.labels = labels

        if shuffle:
            indices = list(range(len(self.paths)))
            random.shuffle(indices)
            self.paths = [self.paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Return the batch of data at <index>.
        """
        path = self.paths[index]
        data = torch.from_numpy(np.load(path))
        return data, self.labels[index]

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.labels)


def construct_datasets(inputs: List[str], labels: List[int], train_split: float,
                       val_split: float) -> Tuple[MFCCDataset, MFCCDataset, MFCCDataset]:
    """
    Split the dataset based on the train and validation split designated and return the datasets.
    """
    indices = list(range(len(inputs)))
    random.shuffle(indices)

    inputs = [inputs[i] for i in indices]
    labels = [labels[i] for i in indices]

    train_size = int(np.ceil(len(inputs) * train_split))
    val_size = int(np.ceil(len(inputs) * val_split))
    test_size = len(inputs) - train_size - val_size

    training_data = inputs[: train_size]
    training_labels = labels[: train_size]
    validation_data = inputs[train_size: train_size + val_size]
    validation_labels = labels[train_size: train_size + val_size]
    testing_data = inputs[len(inputs) - test_size:]
    testing_labels = labels[len(inputs) - test_size:]

    return MFCCDataset(training_data, training_labels), \
        MFCCDataset(validation_data, validation_labels), \
        MFCCDataset(testing_data, testing_labels)
