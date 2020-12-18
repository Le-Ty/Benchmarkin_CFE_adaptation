## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import load_model

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import library.data_processing as processing

class Data_Tabular:
	def __init__(self, data_path, data_name, target_name, continuous_features):

		data = pd.read_csv(data_path + data_name)
		columns = data.columns
		cat_features = processing.get_categorical_features(columns, continuous_features, target_name)

		data_one_hot = processing.one_hot_encode_instance(data, data, cat_features)
		y = data_one_hot[target_name]
		data_one_hot = data_one_hot.drop(columns=[target_name])

		one_hot_encoder = OneHotEncoder(sparse=False)
		integer_encoded = y.values.reshape(len(y), 1)
		y = one_hot_encoder.fit_transform(integer_encoded)
		data_normalized = normalize_instance(data_one_hot, data_one_hot, continuous_features)

		train_data, self.test_data, train_labels, self.test_labels = train_test_split(data_normalized.values, y, train_size=0.8)

		VALIDATION_SIZE = 2000

		self.validation_data = train_data[:VALIDATION_SIZE, :]
		self.validation_labels = train_labels[:VALIDATION_SIZE]
		self.train_data = train_data[VALIDATION_SIZE:, :]
		self.train_labels = train_labels[VALIDATION_SIZE:]



class Model_Tabular:
	def __init__(self, dim_input, dim_hidden_layer1, dim_hidden_layer2, dim_output_layer, num_of_classes,
				 restore=None, session=None, use_log=False):
		'''
		:param dim_input: int > 0, number of neurons for this layer (for Adult: 104)
		:param dim_hidden_layer_1: int > 0, number of neurons for this layer (for Adult: 30)
		:param dim_hidden_layer_2: int > 0, number of neurons for this layer (for Adult: 15)
		:param dim_output_layer: int > 0, number of neurons for this layer (for Adult: 5)
		:param num_of_classes: int > 0, number of classes (for Adult: 2)
		'''
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

		# output log probability, used for black-box attack
		if use_log:
			model.add(Activation('sigmoid'))
		if restore:
			model.load_weights(restore)
			model.summary()

		self.model = model

	def predict(self, data):
		return self.model(data)