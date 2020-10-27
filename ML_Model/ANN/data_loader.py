import torch

import torch.utils.data as data
import pandas as pd
import library.data_processing as processing


class DataLoader(data.Dataset):
    def __init__(self, path, label, normalization=False):
        """
        Load training dataset
        :param path: string with path to training set
        :param label: string, column name for label
        :return: tensor with trainingdata
        """
        # load dataset
        self.dataset = pd.read_csv(path)
        self.target = label

        if normalization:
            self.dataset = processing.normalize(self.dataset, label)

        # Save target and predictors
        self.X = self.dataset.drop(self.target, axis=1)
        self.y = self.dataset[self.target]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # select correct row with idx
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]