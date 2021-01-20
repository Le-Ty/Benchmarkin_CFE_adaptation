from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Multiply
from keras.models import Model, Sequential


class Train_AE:
	def __init__(self, dim_input, dim_hidden_layer1, dim_hidden_layer2, dim_latent, data_name,
				 epochs=25, learning_rate=0.002, batch_size=64):

		"""
		Defines the structure of the autoencoder
		:param dim_input: int > 0; number of neurons for this layer (for Adult: 104)
		:param dim_hidden_layer_1: int > 0, number of neurons for this layer (for Adult: 30)
		:param dim_hidden_layer_2: int > 0, number of neurons for this layer (for Adult: 15)
		:param dim_latent: int >0; number of dimensions for the latent code
		:param learning_rate: float > 0; learning rate (for Adult: 0.002)
		:param epochs: int > 0; number of epochs (for Adult: 60)
		:batch size: int > 0; batch_size (for Adult: 60)
		"""

		self.dim_input = dim_input
		self.dim_latent = dim_latent
		self.dim_hidden_layer1 = dim_hidden_layer1
		self.dim_hidden_layer2 = dim_hidden_layer2
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.data_name = data_name


	def Build_Train_Save_Model(self, xtrain, ytrain, xtest, ytest):

		def loss(y_true, y_pred):
			""" Negative log likelihood (Bernoulli). """

			# Works if data is normalized between 0 and 1!
			# Keras.losses.binary_crossentropy gives the mean
			# Over the last axis. we require the sum

			return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

		x = Input(shape=(self.dim_input,))

		# Encoder
		encoded = Dense(self.dim_hidden_layer1, activation='relu')(x)
		encoded = Dense(self.dim_hidden_layer2, activation='relu')(encoded)
		z = Dense(self.dim_latent, activation='relu')(encoded)

		# Decoder
		decoder = Sequential([
			Dense(self.dim_hidden_layer1, input_dim=self.dim_latent, activation='relu'),
			Dense(self.dim_hidden_layer2, activation='relu'),
			Dense(self.dim_input, activation='sigmoid')
			])

	# Compile Autoencoder, Encoder & Decoder
		encoder = Model(x, z)
		xhat = decoder(z)
		autoencoder = Model(x, xhat)
		autoencoder.compile(optimizer='rmsprop', loss=loss)

	# Train model
		autoencoder.fit(xtrain,
					xtrain,
					shuffle=True,
					epochs=self.epochs,
					batch_size=self.batch_size,
					validation_data=(xtest, xtest))


		# save ae weights
		model_name = 'ae_tf'

		autoencoder.save_weights('CF_Models/cem_ml/AE_models/Saved_Models/{}_{}.{}'.format(model_name, self.data_name, 'h5'))

		# save model
		model_json = autoencoder.to_json()

		with open('CF_Models/cem_ml/AE_models/Saved_Models/{}_{}.{}'.format(model_name, self.data_name, 'json'), "w") as json_file:
			json_file.write(model_json)
	
		# return [autoencoder, encoder, decoder]