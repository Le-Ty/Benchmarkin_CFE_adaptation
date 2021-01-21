import torch

import numpy as np
import pandas as pd
import library.data_processing as processing

from torch import nn

import torch

import numpy as np
import pandas as pd
import library.data_processing as processing

from torch import nn


class Linear(nn.Module):
    def __init__(self, input_layer, num_of_classes):
        """
        Defines the structure of the neural network
        :param input_layer: int > 0, number of neurons for this layer
        :param output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        """
        super().__init__()

        # number of input neurons
        self.input_neurons = input_layer

        # Layer
        self.layer = nn.Linear(input_layer, num_of_classes)

    def forward(self, x):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        output = self.layer(x)

        # output = output.squeeze()

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

        class_1 = 1 - self.forward(input).detach().numpy().squeeze()
        class_2 = self.forward(input).detach().numpy().squeeze()

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

    def get_coefficients(self):
        """
        Calculates the coefficients of the model.
        :return: Tuple of tensor (coefficients), tensor (intersection)
        """
        coef = self.layer.weight.data
        inter = self.layer.bias.data

        return coef, inter


