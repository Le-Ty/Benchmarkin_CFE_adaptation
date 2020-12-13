import torch

import numpy as np
import pandas as pd
import library.data_processing as processing

from torch import nn


class ANN(nn.Module):
    def __init__(self, input_layer, hidden_layer_1, hidden_layer_2, output_layer, num_of_classes):
        """
        Defines the structure of the neural network
        :param input_layer: int > 0, number of neurons for this layer
        :param hidden_layer_1: int > 0, number of neurons for this layer
        :param hidden_layer_2: int > 0, number of neurons for this layer
        :param output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        """
        super().__init__()

        # number of input neurons
        self.input_neurons = input_layer

        # Layer
        self.input = nn.Linear(input_layer, hidden_layer_1)
        self.hidden_1 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.hidden_2 = nn.Linear(hidden_layer_2, output_layer)
        self.output = nn.Linear(output_layer, num_of_classes)

        # Activation
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        output = self.input(x)
        output = self.relu(output)
        output = self.hidden_1(output)
        output = self.relu(output)
        output = self.hidden_2(output)
        output = self.relu(output)
        output = self.output(output)
        output = self.sigmoid(output)

        output = output.squeeze()

        return output

    def prob_predict(self, data):
        """
        Computes probabilistic output for two classes
        :param data: torch tabular input
        :return: np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)

        class_1 = 1 - self.forward(input).detach().numpy()
        class_2 = self.forward(input).detach().numpy()

        # For single prob prediction it happens, that class_1 is casted into float after 1 - prediction
        # Additionally class_1 and class_2 have to be at least shape 1
        if not isinstance(class_1, np.ndarray):
            class_1 = np.array(class_1).reshape(1)
            class_2 = class_2.reshape(1)

        return np.array(list(zip(class_1, class_2)))

    def predict(self, data):
        """
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)

        return self.forward(input).detach().numpy()


