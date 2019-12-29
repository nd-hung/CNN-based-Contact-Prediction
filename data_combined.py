import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

MAP_CHANNELS = 60
RAW_CHANNELS = 441


# define PSICOV150 test dataset
class Psicov150(Dataset):
    """
    Test Dataset for combined LSTM-CNN contact prediction model.
    CNN Features are extracted by cov21stats, at https://github.com/psipred/DeepCov
    """

    def __init__(self, root_dir, target_list, seq_file):
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
            seq_file (csv file): File contains test target name and sequence
        """
        self.root_dir = root_dir
        self.target_list = target_list
        self.test_sequences = self.read_test_sequences(seq_file)

    def read_test_sequences(self, seq_file):
        test_seq_df = pd.read_csv(seq_file, delimiter=' ',
                                  header=None, names=['target', 'seq'])
        return test_seq_df

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
        # aln_file = os.path.join(self.root_dir, 'aln', target+'.aln')
        # with open(aln_file) as f:
        #     sequence = f.readline()
        #     sequence = sequence.strip()
        df = self.test_sequences[test_sequences['target'] == target+'.pdb']
        sequence = df['seq'].values.tolist()[0]
        sample = {'target': target, 'cov': target_cov, 'sequence': sequence}

        return sample
