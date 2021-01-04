import pandas as pd
import numpy as np
import CF_Models.growing_spheres_ml.gs_counterfactuals as gs_ml
import library.data_processing as preprocessing


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
	
	# normalize instances
	instances = preprocessing.normalize_instance(data, instances, continuous_cols)
	
	# binarize instances in robust way
	instances = preprocessing.robust_binarization(instances, binary_cols, continuous_cols)
	
	if dataset_name == 'adult':
	
		counterfactuals = []
		
		# choose mutabale vs. immutable
		keys_correct = continuous_cols + binary_cols
		
		# these keys chosen for sake of illustration
		keys_immutable = ['age', 'sex']
		keys_mutable = list(set(keys_correct) - set(keys_immutable))
	
	
		for i in range(instances.shape[0]):
			
			instance = instances.iloc[i]
			counterfactual = gs_ml.growing_spheres_search(instance, keys_mutable, keys_immutable,
															 continuous_cols, binary_cols, model)
			
			counterfactuals.append(counterfactual)
	
		counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
		counterfactuals_df.columns = instances.columns
	
	return instances, counterfactuals_df
	
