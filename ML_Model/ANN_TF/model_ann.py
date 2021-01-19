import time

import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input, Dense, Lambda, Layer, Multiply, Activation
from keras.models import Model, Sequential
from keras import activations
from keras import optimizers


class Train_ANN():
    def __init__(self, dim_input, dim_hidden_layer1, dim_hidden_layer2, dim_output_layer, num_of_classes, data_name,
                 epochs=10, learning_rate=0.002, batch_size=64):
        """
        Defines the structure of the neural network
        :param dim_input: int > 0, number of neurons for this layer
        :param dim_hidden_layer_1: int > 0, number of neurons for this layer
        :param dim_hidden_layer_2: int > 0, number of neurons for this layer
        :param dim_output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes (e.g. for Adult: 2)
        :param learning_rate: int > 0, learning rate (e.g. for Adult: 0.002)
        :param epochs: int > 0; number of epochs (e.g. for Adult: 10)
        :param batch_size: int > 0; batch_size (e.g. for Adult: 64)

        """
        self.dim_input = dim_input
        self.dim_hidden_layer1 = dim_hidden_layer1
        self.dim_hidden_layer2 = dim_hidden_layer2
        self.dim_output_layer = dim_output_layer
        self.num_of_classes = num_of_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_name = data_name

    def Build_Train_Save_Model(self, xtrain, ytrain, xtest, ytest):
        model = Sequential([Dense(self.dim_hidden_layer1, input_dim=self.dim_input, activation='relu'),
                            Dense(self.dim_hidden_layer2, activation='relu'),
                            Dense(self.dim_output_layer, activation='relu'),
                            Dense(self.num_of_classes, activation='softmax')])

        # sgd = optimizers.SGD(lr=self.learning_rate, momentum=0.9, decay=0, nesterov=False)

        # Compile the model
        model.compile(
            optimizer='rmsprop',  # works better than sgd
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # Train the model
        model.fit(
            xtrain,
            to_categorical(ytrain),
            epochs=self.epochs,
            shuffle=True,
            batch_size=self.batch_size,
            validation_data=(xtest, to_categorical(ytest)))

        hist = model
        test_error = (1 - hist.history.history['val_acc'][-1])

        # save model
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        model_name = 'ann_tf'
        model.save('ML_Model/Saved_Models/ANN_TF/{}_{}_input_{:.0f}'.format(model_name, self.data_name, self.dim_input))


class Model_Tabular:
    def __init__(self, dim_input, dim_hidden_layer1, dim_hidden_layer2, dim_output_layer, num_of_classes,
                 restore=None, session=None, use_prob=False):

        # For model loading
        """
        :param dim_input: int > 0, number of neurons for this layer
        :param dim_hidden_layer_1: int > 0, number of neurons for this layer
        :param dim_hidden_layer_2: int > 0, number of neurons for this layer
        :param dim_output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        :param use_prob: boolean; FALSE required for CEM; all others should use True
        """  #

        self.dim_input = dim_input
        self.dim_hidden_layer1 = dim_hidden_layer1
        self.dim_hidden_layer2 = dim_hidden_layer2
        self.dim_output_layer = dim_output_layer
        self.num_of_classes = num_of_classes

        model = Sequential([
            Dense(self.dim_hidden_layer1, input_dim=self.dim_input, activation='relu'),
            Dense(self.dim_hidden_layer2, activation='relu'),
            Dense(self.dim_output_layer, activation='relu'),
            Dense(self.num_of_classes)
        ])

        # whether to output probability
        if use_prob:
            model.add(Activation(activations.softmax))
        if restore:
            model.load_weights(restore)
            model.summary()

        self.model = model

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        return self.model(data)
