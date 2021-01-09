import pandas as pd
import numpy as np
import torch
import CF_Models.clue_ml.Clue_model.CLUE_counterfactuals as clue_ml
import CF_Models.clue_ml.AE_models.AE.vae_training as vae_training
from CF_Models.clue_ml.AE_models.AE.fc_gauss_cat import VAE_gauss_cat_net

from sklearn.model_selection import train_test_split
import library.data_processing as preprocessing
import timeit

from os import path

def get_counterfactual(dataset_path, dataset_filename, dataset_name,
					   instances, binary_cols, continuous_cols, target_name, model, train_vae=False):
	"""
	:param dataset_path: str; path
	:param dataset_filename: str; Filename (e.g. 'adult_full')
	:param dataset_name: str; (e.g. 'adult')
	:param instances: pd data frame; Instances to generate counterfactuals for
	:param binary_cols: list; list of features names, which need to be one_hot_encoded
	:param continuous_cols: list; list of numeric features names
	:param target_name: str; target corresponding to data set
	:param model; classification model (either tf keras, pytorch or sklearn)
	:param train_vae; boolean; enforce VAE training (also works if VAE already trained, and could already be loaded)
	:return: input instances & counterfactual explanations
	"""  #
	
	# indicates whether we use the model with one-hot-encoded features; if False, one-hot encoded features
	ann_binary = False
	
	# Authors say: 'For automatic explainer generation'
	flat_vae_bools = False
	
	# VAE model archiceture parameters
	widths = [10, 10, 10, 10]
	depths = [3, 3, 3, 3]  # Authors go deeper because they are using residual models
	latent_dims = [6, 8, 4, 4]
	
	# drop targets
	data = pd.read_csv(dataset_path + dataset_filename)
	data = data.drop(columns=[target_name])
	instances = instances.drop(columns=[target_name])
	
	
	# normalize data
	data_processed = preprocessing.normalize_instance(data, data, continuous_cols)
	# binarize cat binary instances in robust way: here we >>do not want<< to drop first column
	data_processed = preprocessing.robust_binarization(data_processed, binary_cols, continuous_cols, drop_first=False)
	
	# normalize instances
	instances = preprocessing.normalize_instance(data, instances, continuous_cols)
	if ann_binary:
		# binarize cat binary instances in robust way
		instances = preprocessing.robust_binarization(instances, binary_cols, continuous_cols, drop_first=True)
	else:
		instances = preprocessing.one_hot_encode_instance(data, instances, binary_cols)
	
	if dataset_name == 'adult':
		
		# choose mutabale vs. immutable
		keys_correct = continuous_cols + binary_cols
		keys_immutable = ['sex']
		keys_mutable = list(set(keys_correct) - set(keys_immutable))
	
	elif dataset_name == 'compas':
		print('not considered yet')
	
	elif dataset_name == 'GiveMeSomeCredit':
		print('not considered yet')
	
	elif dataset_name == 'heloc':
		print('not considered yet')
	
	# STEP 1: VAE training + model saving, if no model has been saved yet
	width = widths[dataset_name.index(dataset_name)]
	depth = depths[dataset_name.index(dataset_name)]  # number of hidden layers
	latent_dim = latent_dims[dataset_name.index(dataset_name)]
	
	# normalize and split test & train data for VAE
	data_normalized = preprocessing.normalize_instance(data_processed, data_processed, continuous_cols)
	x_train, x_test = train_test_split(data_normalized.values, train_size=0.7)
	
	# Error message when training VAE using float 64: -> Change to: float 32
	# "Expected object of scalar type Float but got scalar type Double for argument #2 'mat1' in call to _th_addmm"
	x_train = np.float32(x_train)
	x_test = np.float32(x_test)

	# indicate dimensions of inputs -- input_dim_vec: (if binary = 2; if continuous = 1)
	input_dims_continuous = list(np.repeat(1, len(continuous_cols)))
	input_dims_binary = list(np.repeat(2, len(binary_cols)))
	input_dim_vec = input_dims_continuous + input_dims_binary
	
	# check directory for VAE-weights file
	check_dir = 'C:/Users/fred0/Documents/proj/Benchmarkin_Counterfactual_Examples/CF_Models/clue_ml/AE_models/Saved_models/fc_VAE_' + dataset_name + '_models/theta_best.dat'
	save_dir = 'C:/Users/fred0/Documents/proj/Benchmarkin_Counterfactual_Examples/CF_Models/clue_ml/AE_models/Saved_models/fc_VAE_' + dataset_name
	
	if path.isfile(check_dir):
		if train_vae:
			# train and save VAE if not trained yet
			vae_training.training(x_train, x_test, input_dim_vec, dataset_name, width, depth, latent_dim)
	else:
		# train and save VAE if not trained yet
		vae_training.training(x_train, x_test, input_dim_vec, dataset_name, width, depth, latent_dim)
	
	cuda = torch.cuda.is_available()
	VAE = VAE_gauss_cat_net(input_dim_vec, width, depth, latent_dim, pred_sig=False,
							lr=1e-4, cuda=cuda, flatten=flat_vae_bools)
	VAE.load(save_dir + '_models/theta_best.dat')


	# STEP 2: for every instance 'under consideration', use CLUE to find counterfactual
	counterfactuals = []
	times_list = []
	
	for index in range(instances.shape[0]):
		start = timeit.default_timer()
		counterfactual = clue_ml.vae_gradient_search(instances.values[index, :], model, VAE)
		stop = timeit.default_timer()
		time_taken = stop - start
		
		times_list.append(time_taken)
		counterfactuals.append(counterfactual)
	
	counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
	counterfactuals_df.columns = instances.columns
	
	# order counterfactuals and instances in original data order
	counterfactuals_df = counterfactuals_df[data.columns]
	instances = instances[data.columns]
	
	# convert binary cols of counterfactuals and instances into strings
	counterfactuals_df[binary_cols] = counterfactuals_df[binary_cols].astype("string")
	instances[binary_cols] = instances[binary_cols].astype("string")
	
	return instances, counterfactuals_df, times_list