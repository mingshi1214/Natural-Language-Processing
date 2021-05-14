import os
from typing import Tuple, List
import torch
from random import choice


Tensor = torch.Tensor


def pool_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Pool all the numpy MFCC files into 1 big numpy array. <path> is the directory containing all the subdirectories
    with the data within.

    Return the data and the corresponding labels.
    """
    inputs = []
    labels = []
    for item in os.listdir(path):
        if not os.path.isdir(os.path.join(path, item)):
            continue

        subdir_path = os.path.join(path, item)
        mfcc_files = [file for file in os.listdir(subdir_path) if file.endswith("npy")]
        mfcc_files.sort(key=lambda x: int(x[0]))
        mfcc_files = [os.path.join(subdir_path, file) for file in mfcc_files]

        inputs.extend(mfcc_files)

        with open(os.path.join(subdir_path, "transcripts.Kaldi.txt")) as transcript:
            lines = transcript.readlines()
            for line in lines:
                tag = line.strip().split()[1]
                label = tag[:tag.find("/")]
                # both lie up and lie down are considered lie
                label = 0 if label == "T" else 1    # lie detection so 0 if truth, 1 if lie
                labels.append(label)

    return inputs, labels


def pad_batch(inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Pad (with 0) the inputs to have the same length.
    """
    all_lengths = [item.size(0) for item in inputs]
    max_length = max(all_lengths)
    data_dim = inputs[0].size(1)
    padded = []

    for item in inputs:
        num_rows = max_length - item.size(0)
        padding = torch.zeros(num_rows, data_dim, dtype=item.dtype)
        padded.append(torch.cat([item, padding]))

    return torch.stack(padded), torch.LongTensor(all_lengths)


def pad_and_sort_batch(elements: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Pad the batch of data and return the data, lengths and labels in decreasing length order.
    """
    inputs = []
    labels = []

    for element in elements:
        inputs.append(element[0])
        labels.append(element[1])

    padded_inputs, inputs_lengths = pad_batch(inputs)
    sorted_indices = inputs_lengths.topk(len(inputs_lengths))[1]

    padded_inputs = padded_inputs[sorted_indices]
    inputs_lengths = inputs_lengths[sorted_indices]
    labels = torch.LongTensor(labels)[sorted_indices]

    return padded_inputs, inputs_lengths, labels
