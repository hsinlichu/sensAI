import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FordADataset(Dataset):
    def __init__(self, path, train, mean=0, std=1):
        self.data = pd.read_csv(path, header=None, sep='\t')
        print(self.data.head())
        self.data.loc[:, 0] = (self.data.loc[:, 0] == 1)
        self.data = self.data.to_numpy().astype(np.float32)
        if train:
            self.mean = self.data[:, 1:].mean()
            self.std = self.data[:, 1:].std()
        else:
            self.mean = mean
            self.std = std

        print("Mean: {} Std: {}".format(self.mean, self.std))
        self.data[:, 1:] = (self.data[:, 1:] - self.mean) / self.std
        print(self.data[:, 1:].mean(), self.data[:, 1:].std())
        print(self.data)
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx, 1:]
        labels = self.data[idx, 0]
        labels = labels.astype(np.int64)
        inputs = inputs[:, np.newaxis]

        return inputs, labels
