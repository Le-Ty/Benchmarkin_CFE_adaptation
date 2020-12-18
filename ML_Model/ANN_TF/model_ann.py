import os
import time

from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Multiply
from keras.models import Model, Sequential
from keras import optimizers
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Train_ANN():
	def __init__(self, dim_input, dim_hidden_layer1, dim_hidden_layer2, dim_output_layer, num_of_classes, data_name,
				 epochs=10, learning_rate=0.002, batch_size=64):
		"""
		Defines the structure of the neural network
		:param dim_input: int > 0, number of neurons for this layer (for Adult: 104)
		:param dim_hidden_layer_1: int > 0, number of neurons for this layer (for Adult: 30)
		:param dim_hidden_layer_2: int > 0, number of neurons for this layer (for Adult: 15)
		:param dim_output_layer: int > 0, number of neurons for this layer (for Adult: 5)
		:param num_of_classes: int > 0, number of classes (for Adult: 2)
		:param learning_rate: int > 0, learning rate (for Adult: 0.002)
		:param epochs: int > 0; number of epochs (for Adult: 10)
		:param batch_size: int > 0; batch_size (for Adult: 64)

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

		#sgd = optimizers.SGD(lr=self.learning_rate, momentum=0.9, decay=0, nesterov=False)

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
		model.save('C:/Users/fred0/Documents/proj/Benchmarkin_Counterfactual_Examples/ML_Model/Saved_Models/ANN_TF/{}_{}_input_{:.0f}'.format(model_name, self.data_name, self.dim_input))
