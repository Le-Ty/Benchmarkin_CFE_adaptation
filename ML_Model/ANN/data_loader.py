import torch
import pickle

import torch.utils.data as data
import pandas as pd
import library.data_processing as processing


class DataLoader(data.Dataset):
    def __init__(self, path, filename, label, normalization=False, encode=False):
        """
        Load training dataset
        :param path: string with path to training set
        :param label: string, column name for label
        :return: tensor with trainingdata
        """
        # load dataset
        self.dataset = pd.read_csv(path + filename)
        self.target = label

        # load categorical feature names
        with open(path + 'cat_names.txt', 'rb') as f:
            self.categorical_features = pickle.load(f)

        if encode:
            self.dataset = pd.get_dummies(self.dataset, columns=self.categorical_features)

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

    def get_number_of_features(self):
        return self.X.shape[1]

    def drop(self, columns, axis=0):
        """
        Similar to pandas method, it drops specific columns for a given axis.
        :param columns: List with column names to drop
        :param axis: specific axis to drop
        :return: No return value
        """

        self.dataset = self.dataset.drop(columns, axis=axis)

        # Save target and predictors
        self.X = self.dataset.drop(self.target, axis=1)
        self.y = self.dataset[self.target]
