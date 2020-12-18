import torch
import time
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn import preprocessing

import ML_Model.ANN.model as model
import ML_Model.ANN.data_loader as loader
import library.data_processing as processing


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
               '../../Saved_Models/ANN/{}_input_{}_lr_{}_te_{:.2f}.pt'.format(timestamp, model.input_neurons, learning_rate, test_error))


# Introducing a small ANN model, trained on few features to test runtime of MACE
small_model = False

if not small_model:
    # Dataloader
    dataset = loader.DataLoader('../Datasets/Adult/',
                                'adult_full.csv', 'income', normalization=True, encode=True)

    # Split into train and test set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

    # Define the model
    input_size = dataset.get_number_of_features()
    model = model.ANN(input_size, 18, 9, 3, 1)

    trainloader = DataLoader(dataset_train, batch_size=100, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=100, shuffle=False)

    training(model, trainloader, testloader, 0.002, 100)
else:
    # Dataloader / Normalization OFF for MACE
    dataset = loader.DataLoader('../Datasets/Adult/',
                                'adult_full.csv', 'income', normalization=False, encode=False)

    # Process dataset to get very few features
    dataset.drop(['marital-status', 'workclass', 'occupation', 'relationship'], axis=1)
    # Get values we want to replace
    df = dataset.dataset
    to_replace_in_race = list(dict.fromkeys(df['race'].values.tolist()))
    to_replace_in_race.remove('White')

    to_replace_in_native_country = list(dict.fromkeys(df['native-country'].values.tolist()))
    to_replace_in_native_country.remove('US')

    # replace values
    dataset.replace('race', to_replace_in_race, 'non-white')
    dataset.replace('native-country', to_replace_in_native_country, 'non-US')

    # one-hot-encode dataset
    dataset.one_hot_encode()
    
    # Split into train and test set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

    # Define the model
    input_size = dataset.get_number_of_features()
    model = model.ANN(input_size, 8, 4, 2, 1)

    trainloader = DataLoader(dataset_train, batch_size=100, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=100, shuffle=False)

    training(model, trainloader, testloader, 0.002, 100)
