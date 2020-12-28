import pandas as pd
import numpy as np
import CF_Models.face_ml.face_counterfactuals as face_ml
import library.data_processing as preprocessing


def get_counterfactual(dataset_path, dataset_filename, dataset_name,
					   instances, binary_cols, continuous_cols, target_name, model, mode='knn'):
	"""
	:param dataset_path: str; path
	:param dataset_filename: str; Filename (e.g. 'adult_full')
	:param dataset_name: str; (e.g. 'adult')
	:param instances: pd data frame; Instances to generate counterfactuals for
	:param binary_cols: list; list of features names, which need to be one_hot_encoded
	:param continuous_cols: list; list of numeric features names
	:param target_name: str; target corresponding to data set
	:param model; classification model (either tf keras, pytorch or sklearn)
	:param mode: str; mode (either 'knn' or 'epsilon')
	:return: input instances & counterfactual explanations
	"""  #
	
	# drop targets
	data = pd.read_csv(dataset_path + dataset_filename)
	data = data.drop(columns=[target_name])
	instances = instances.drop(columns=[target_name])
	
	
	# normalize data
	data_processed = preprocessing.normalize_instance(data, data, continuous_cols)
	# binarize cat binary instances in robust way
	data_processed = preprocessing.robust_binarization(data_processed, binary_cols, continuous_cols)
	
	print(data_processed.shape)
	
	# normalize instances
	instances = preprocessing.normalize_instance(data, instances, continuous_cols)
	# binarize cat binary instances in robust way
	instances = preprocessing.robust_binarization(instances, binary_cols, continuous_cols)
	
	if dataset_name == 'adult':
		
		counterfactuals = []
		
		# choose mutabale vs. immutable
		keys_correct = continuous_cols + binary_cols
		keys_immutable = ['sex']
		keys_mutable = list(set(keys_correct) - set(keys_immutable))
		# keys_mutable_and_immutable = keys_mutable + keys_immutable
		
	elif dataset_name == 'compas':
		print('not considered yet')
		
	elif dataset_name == 'GiveMeSomeCredit':
		print('not considered yet')

	elif dataset_name == 'heloc':
		print('not considered yet')
		
	# drop instances under consideration & reorder
	cond = data_processed.isin(instances).values
	data_processed = data_processed.drop(data_processed[cond].index)
	data_processed_ordered = pd.concat([instances, data_processed], ignore_index=True)
	
	for index in range(instances.shape[0]):
		
		counterfactual = face_ml.graph_search(data_processed_ordered, index, keys_mutable, keys_immutable,
											  continuous_cols, binary_cols, model, mode=mode)
		counterfactuals.append(counterfactual)
		
	counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
	counterfactuals_df.columns = instances.columns
	
	return instances, counterfactuals_df

