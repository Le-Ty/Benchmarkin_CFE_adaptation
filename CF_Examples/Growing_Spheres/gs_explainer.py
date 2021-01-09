import pandas as pd
import numpy as np
import CF_Models.growing_spheres_ml.gs_counterfactuals as gs_ml
import library.data_processing as preprocessing
import timeit


def get_counterfactual(dataset_path, dataset_filename, dataset_name,
					   instances, binary_cols, continuous_cols, target_name, model):
	"""
	:param dataset_path: str; path
	:param dataset_filename: str; filename (e.g. 'adult_full')
	:param dataset_name: str; (e.g. 'adult')
	:param instances: pd data frame; instances to generate counterfactuals for
	:param binary_cols: list; list of features names, which need to be one_hot_encoded
	:param continuous_cols: list; list of numeric features names
	:param target_name: str; target corresponding to data set
	:param model; classification model (either tf keras, pytorch or sklearn)
	:return: input instances & counterfactual explanations
	"""  #
	
	# drop targets
	data = pd.read_csv(dataset_path + dataset_filename)
	data = data.drop(columns=[target_name])
	instances = instances.drop(columns=[target_name])
	
	instances_a = instances
	
	# normalize instances
	instances = preprocessing.normalize_instance(data, instances, continuous_cols)
	
	# binarize instances in robust way
	instances = preprocessing.robust_binarization(instances, binary_cols, continuous_cols)
	
	if dataset_name == 'adult':
		
		# choose mutabale vs. immutable
		keys_correct = continuous_cols + binary_cols
		
		# these keys chosen for sake of illustration
		keys_immutable = ['age', 'sex']
		keys_mutable = list(set(keys_correct) - set(keys_immutable))
	
	counterfactuals = []
	times_list = []
	
	for i in range(instances.shape[0]):
			
		instance = instances.iloc[i]
		start = timeit.default_timer()
		counterfactual = gs_ml.growing_spheres_search(instance, keys_mutable, keys_immutable,
															 continuous_cols, binary_cols, model)
		stop = timeit.default_timer()
		time_taken = stop - start
			
		counterfactuals.append(counterfactual)
		times_list.append(time_taken)
	
	counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
	counterfactuals_df.columns = instances.columns
		
	# Obtain labels
	instance_label = np.argmax(model.model.predict(instances.values), axis=1)
	counterfactual_label = np.argmax(model.model.predict(counterfactuals_df.values), axis=1)
		
	# Round binary columns to integer
	counterfactuals_df[binary_cols] = counterfactuals_df[binary_cols].round(0).astype(int)
		
	# Order counterfactuals and instances in original data order
	counterfactuals_df = counterfactuals_df[data.columns]
	instances = instances[data.columns]
		
	# Convert binary cols of counterfactuals and instances into strings: Required for >>Measurement<< in script
	counterfactuals_df[binary_cols] = counterfactuals_df[binary_cols].astype("string")
	instances[binary_cols] = instances[binary_cols].astype("string")
		
	# Convert binary cols back to original string encoding
	counterfactuals_df = preprocessing.map_binary_backto_string(data, counterfactuals_df, binary_cols)
	instances = preprocessing.map_binary_backto_string(data, instances, binary_cols)

	# Add labels
	counterfactuals_df[target_name] = counterfactual_label
	instances[target_name] = instance_label
		
	# Collect in list making use of pandas
	instances_list = []
	counterfactuals_list = []
	for i in range(counterfactuals_df.shape[0]):
		counterfactuals_list.append(pd.DataFrame(counterfactuals_df.iloc[i].values.reshape((1, -1)), columns=counterfactuals_df.columns))
		instances_list.append(pd.DataFrame(instances.iloc[i].values.reshape((1, -1)), columns=instances.columns))
			
	return instances_list, counterfactuals_list, times_list
	
