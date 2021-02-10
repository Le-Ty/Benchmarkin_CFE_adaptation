import sys
import pandas as pd

sys.path.append('../..')

from CF_Examples.counterfact_expl.CE.experiments.run_experiments import run_experiments
from CF_Examples.counterfact_expl.CE.path import get_path
import keras

'''
runs the synthetic experiment. Results are saved in the outputs/synthetic_test directory.
'''
#initializing
path = get_path()
classifier_name = "ANN"
dataset = "Adult"
#read in query
query_instances = pd.read_csv("CF_Input/" + dataset + "/" + classifier_name + "/query_instances.csv", index_col = False)

#immutable features for each dataset
im_feat_adult = ["age", "sex"]
im_feat_gmc = ["age", "NumberOfDependents"]
im_feat_compas = ["age", "sex"]

keras.backend.clear_session()

query, cf, direct_cost, time, success_rate = run_experiments(
			im_feat = im_feat_adult, query = query_instances.head(6),
			train_steps = 8000, model = classifier_name, dataset = dataset,
			train_AAE = False
			)
