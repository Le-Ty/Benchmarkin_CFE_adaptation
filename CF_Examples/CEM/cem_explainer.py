import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import time
from CF_Models.cem_ml.setup_data_model import Data_Tabular, Model_Tabular
import CF_Models.cem_ml.utils as util
from CF_Models.cem_ml.aen_cem import AEADEN
import library.data_processing as preprocessing


def counterfactual_search(dataset_filename, instance, model=None, max_iter=100,
                       binary_steps=9, init_const=10, mode='PN', kappa=0, beta=1e-1, gamma=0.5):
	tf.reset_default_graph()
	
	with tf.Session() as sess:
		"""
		Compute counterfactual for CEM
		:param dataset_filename: String; Filename (e.g. 'adult_full')
		:param instances: Dataframe; Instances to generate counterfactuals
		:param model: model; pretrained classifier (could in principle be anything)
		:param mode: String; one of {PN, PP}; PN corresponds to counterfactual explanation
		:param max_iter: int > 0;
		:param init_const: int > 0;
		:param kappa: float (?) > 0; defines the margin between the two classses; the higher kappa, the more certain we will be
		:param beta: float > 0; importance given to ell_1 term
		:param gamma: float > 0; importance given to AutoEncoder term
		"""

		# must match dims of pretrained model (at the moment it matches the pretrained TF ANN adult model in \ML_Model\...
		dim_input = 13
		dim_hidden_layer_1 = 18
		dim_hidden_layer_2 = 9
		dim_output_layer = 3
		num_of_classes = 2
		
		random.seed(121)
		np.random.seed(1211)
	
		data_name = dataset_filename.split('.')[0]
	
		# load the generation model: VAE | AE | AAE
		AE_model = util.load_AE(data_name)
		
		# load the classification model
		model = Model_Tabular(dim_input, dim_hidden_layer_1, dim_hidden_layer_2, dim_output_layer, num_of_classes,
							  restore="ML_Model/Saved_Models/ANN_TF/ann_tf_adult_full_input_13",
							  session=None, use_log=False)
	
		orig_prob, orig_class, orig_prob_str = util.model_prediction(model, np.expand_dims(instance, axis=0))
	
		target_label = orig_class
		orig_sample, target = util.generate_data(instance, target_label)
	
		# start the search
		counterfactual_search = AEADEN(sess, model, mode=mode, AE=AE_model, batch_size=1, kappa=kappa, init_learning_rate=1e-2,
						binary_search_steps=binary_steps, max_iterations=max_iter, initial_const=init_const, beta=beta,
						gamma=gamma)
	
		counterfactual = counterfactual_search.attack(orig_sample, target)
	
		adv_prob, adv_class, adv_prob_str = util.model_prediction(model, counterfactual)
		delta_prob, delta_class, delta_prob_str = util.model_prediction(model, orig_sample - counterfactual)
	
		INFO = "[kappa:{}, Orig class:{}, Adv class:{}, Delta class: {}, Orig prob:{}, Adv prob:{}, Delta prob:{}".format(
			kappa, orig_class, adv_class, delta_class, orig_prob_str, adv_prob_str, delta_prob_str)
		print(INFO)
	
	return instance, counterfactual[0]


def get_counterfactual(data_path, data_name, instances, binary_cols, continuous_features, target_name):
	"""
	:param dataset_filename: str; Filename (e.g. 'adult_full')
	:param instances: Dataframe; Instances to generate counterfactuals for
	:param binary_cols: list; list of features which need to be one_hot_encoded
	:return: input instances & counterfactual explanations
	"""  #
	
	# drop targets
	data = pd.read_csv(data_path + data_name)
	data = data.drop(columns=[target_name])
	instances = instances.drop(columns=[target_name])
	robust_names = list(instances.columns)
	
	# normalize instances
	instances = preprocessing.normalize_instance(data, instances, continuous_features)
	
	# robust processing: when binary feature only contains 1s or 0s, pd.get_dummies does neither one-hot encode
	# properly nor does it binarize properly; thus, we need to make sure that binarization is correct
	instances = pd.get_dummies(instances, prefix_sep="__", columns=binary_cols, drop_first=True)
	non_robust_n = list(instances.columns)
	non_robust_names = []
	for i in range(len(non_robust_n)):
		prefix = non_robust_n[i].split('__')[0]
		non_robust_names.append(prefix)
	instances.columns = non_robust_names
		
	# Add missing columns
	for col in binary_cols:
		if col not in non_robust_names:
			print("Adding missing feature {}".format(col))
			instances[col] = 1

	counterfactuals = []
	
	for i in range(instances.values.shape[0]):
		_, counterfactual = counterfactual_search(data_name, instances.values[i, :])
		counterfactuals.append(counterfactual)
	
	counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
	counterfactuals_df.columns = instances.columns
	
	return instances, counterfactuals_df