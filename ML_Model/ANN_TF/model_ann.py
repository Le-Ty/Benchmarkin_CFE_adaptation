import time

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Multiply, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras import activations
from tensorflow.keras import optimizers
import random
import numpy as np
import shutil

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
        random.seed(121)
        np.random.seed(1211)
        tf.set_random_seed(12111)
        # session = tf.Session()
        session = tf.Session()
        graph = tf.get_default_graph()

        with session.as_default():
            with graph.as_default():
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

                # input_keys_placeholder = tf.placeholder(tf.float32, self.dim_input, 'input_keys_placeholder')
                # output_keys = tf.placeholder(tf.float32, self.dim_output_layer, 'output_keys')

                # save model
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                model_name = 'ann_tf'
                model.save('ML_Model/Saved_Models/ANN_TF/{}_{}_input_{:.0f}'.format(model_name, self.data_name, self.dim_input))





                # if os.path.exists('tensorflow_model'):
                #     shutil.rmtree('tensorflow_model')

                self.model = model
                # session = tf.Session()
                input_keys_placeholder = tf.compat.v1.placeholder(tf.float32, self.dim_input, 'input_keys_placeholder')
                output_keys = tf.compat.v1.placeholder(tf.float32, self.dim_output_layer, 'output_keys')
                # with self.model.graph.as_default():

                tf.compat.v1.saved_model.simple_save(session, 'tensorflow_model',
                                    inputs={"x": input_keys_placeholder},
                                    outputs={"z": output_keys})



        # model


class Model_Tabular:
    def __init__(self, dim_input, dim_hidden_layer1, dim_hidden_layer2, dim_output_layer, num_of_classes,
                 restore=None, use_prob=False):

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


        # tf.global_variables_initializer()
        self.model = model
        session = tf.Session()
        input_keys_placeholder = tf.compat.v1.placeholder(tf.float32, self.dim_input, 'input_keys_placeholder')
        output_keys = tf.compat.v1.placeholder(tf.float32, self.dim_output_layer, 'output_keys')
        # with self.model.graph.as_default():
        # with model.graph.as_default():
        tf.saved_model.simple_save(session, 'laal',
                            inputs={"x": input_keys_placeholder},
                            outputs={"z": output_keys})

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        return self.model(data)

    # def save(self):
        # random.seed(121)
        # np.random.seed(1211)
        # tf.set_random_seed(12111)


    #     #in 20, out 3
    #     input_keys_placeholder = tf.placeholder(tf.float32, self.dim_input, 'input_keys_placeholder')
    #     output_keys = tf.placeholder(tf.float32, self.dim_output_layer, 'output_keys')
    #     self.model.saved_model.simple_save(seesion, 'ANN', inputs={"keys": input_keys_placeholder},outputs={"keys": output_keys})
