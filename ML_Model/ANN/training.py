import torch
import time
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from sklearn import preprocessing

import ML_Model.ANN.model as model
import ML_Model.ANN.data_loader as loader


def training(model, train_loader, test_loader, learning_rate, epochs):
    # Use GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # define the loss
    criterion = nn.MSELoss()

    # declaring optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # training
    loss_per_iter = []
    loss_per_batch = []
    for e in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Training pass
            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        loss_per_batch.append(running_loss / (i + 1))
        print('Epoch {} Loss {:.5f}'.format(e, np.sqrt(running_loss / (i + 1))))

    # Comparing training to test
    dataiter = iter(test_loader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs.float())
    print('==========================================')
    print('Root mean squared error')
    train_error = np.sqrt((loss_per_batch[-1]))
    print('Training:', train_error)
    test_error = np.sqrt(criterion(labels.float(), outputs).detach().cpu().numpy())
    print('Test:', test_error)

    # save model
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    torch.save(model.state_dict(),
               '../Saved_Models/ANN/{}_lr_{}_te_{:.2f}.pt'.format(timestamp, learning_rate, test_error))


# Define the model
model = model.ANN(103, 64, 16, 8, 1)

# Dataloader
dataset_train = loader.DataLoader('../../Datasets/Adult/adult.csv', 'income', normalization=True)
dataset_test = loader.DataLoader('../../Datasets/Adult/adult_test.csv', 'income', normalization=True)

trainloader = DataLoader(dataset_train, batch_size=200, shuffle=True)
testloader = DataLoader(dataset_test, batch_size=200, shuffle=False)

training(model, trainloader, testloader, 0.001, 2000)
