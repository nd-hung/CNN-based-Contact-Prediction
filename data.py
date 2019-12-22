import torch
from torch.utils.data import Dataset
import os
import numpy as np

MAP_CHANNELS = 60
RAW_CHANNELS = 441


class DeepCovDataset(Dataset):
    """
    Dataset for CNN-based contact prediction.
    Features are extracted by cov21stats, at https://github.com/psipred/DeepCov
    """

    def __init__(self, root_dir, target_list):
        """
        Args:
            root_dir (string): Directory of extracted covariance matrices, each corresponds to a target.
            
            E.g.,
            data/
              aln/  # alignment files
              map/  # label contact map files
              21c/  # cov matrices
              test/ # test data

            target_list (list): List of targets.
        """
        self.root_dir = root_dir
        self.target_list = target_list

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target = self.target_list[idx]

        # get (ground truth) map data
        rawdata = np.memmap(os.path.join(self.root_dir, 'map', target + '.map'),
                            dtype=np.float32, mode='r')
        length = int(np.sqrt(rawdata.shape[0] / (MAP_CHANNELS + 1)))
        # convert memmap into array to be Pytorch dataloader compatible
        # use the first matrix as ground truth
        target_map = np.array(rawdata.reshape(MAP_CHANNELS + 1, length, length)[0, :, :])

        # get cov matrix
        target_cov = np.memmap(os.path.join(self.root_dir, '21c', target + '.21c'),
                               dtype=np.float32, mode='r', shape=(RAW_CHANNELS, length, length))
        # convert memmap into array to be Pytorch dataloader compatible
        target_cov = np.array(target_cov)

        sample = {'target': target, 'cov': target_cov, 'map': target_map}

        return sample

# define PSICOV150 test dataset
class Psicov150(Dataset):
    """
    Test Dataset for CNN-based contact prediction.
    Features are extracted by cov21stats, at https://github.com/psipred/DeepCov
    """

    def __init__(self, root_dir, target_list):
        """
        Args:
            root_dir (string): Directory of extracted covariance matrices, each corresponds to a target.

            E.g.,
            psicov150/
              aln/  # alignment files
              pdb/  # ground truth
              21c/  # cov matrices
              rr/   # predicted contacts

            target_list (list): List of targets.
        """
        self.root_dir = root_dir
        self.target_list = target_list

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target = self.target_list[idx]

        # get cov matrix
        target_cov = np.memmap(os.path.join(self.root_dir, '21c', target+'.21c'),
                               dtype=np.float32, mode='r')
        length = int(np.sqrt(target_cov.shape[0] / RAW_CHANNELS))

        target_cov = np.array(target_cov.reshape(1, RAW_CHANNELS, length, length))

        # get first sequence from aln file
        aln_file = os.path.join(self.root_dir, 'aln', target+'.aln')
        with open(aln_file) as f:
            sequence = f.readline()
            sequence = sequence.strip()

        sample = {'target': target, 'cov': target_cov, 'sequence': sequence}

        return sample
