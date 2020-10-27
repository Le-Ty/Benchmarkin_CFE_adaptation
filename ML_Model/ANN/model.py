import torch
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

        # Layer
        self.input = nn.Linear(input_layer, hidden_layer_1)
        self.hidden_1 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.hidden_2 = nn.Linear(hidden_layer_2, output_layer)
        self.output = nn.Linear(output_layer, num_of_classes)

        # Activation
        self.relu = nn.LeakyReLU()
        # self.softmax = nn.Softmax(dim=2)


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
        # output = self.softmax(output)

        return output.squeeze()