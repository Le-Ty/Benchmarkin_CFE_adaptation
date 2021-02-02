import sys
import pandas as pd

sys.path.append('../..')

from CF_Examples.counterfact_expl.CE.experiments.run_synthetic import run_synthetic
from CF_Examples.counterfact_expl.CE.BlackBox.predictor_models import *
from CF_Examples.counterfact_expl.CE.path import get_path
from CF_Examples.counterfact_expl.CE.CFE.LIME import LIME
from CF_Examples.counterfact_expl.CE.CFE.find_cf import find_cf
import keras
import ML_Model.ANN_TF
'''
runs the synthetic experiment. Results are saved in the outputs/synthetic_test directory.
'''
path = get_path()


classifier_name = "Linear"
dataset = "Adult"

#read in query
query_instances = pd.read_csv("CF_Input/" + dataset + "/" + classifier_name + "/query_instances.csv", index_col = False)
# query_instances_a = pd.read_csv("CF_Input/Adult/Linear/query_instances.csv", index_col = False)

# query_instances_c = pd.read_csv("CF_Input/Adult/query_instances.csv", index_col = False).head(10)
#Adult
im_feata = ["age", "sex"]

im_featg = ["age", "NumberOfDependents"]

#COMPAS
im_featc = ["age", "sex"]

keras.backend.clear_session()


run_synthetic(im_feat = im_featc, query = query_instances, train_steps = 8000, model = classifier_name, dataset = dataset, train_AAE = False)
# find_ce()g

#pred = sum_predictor()

#pred.predict()

# plot_shap_vals(shapvals_filepath=path + "outputs/synthetic_test/shap_values.csv",
# 	data_filepath=path + "data/synthetic/sum_synthetic.csv",
# 	indirect_output_filepath=path + "outputs/synthetic_test/indirect_influence_distributions.png",
# 	direct_output_filepath=path + "outputs/synthetic_test/direct_influence_distributions.png",
# 	predictor_filepath=path + "experiments/synthetic/models/sum_predictor.h5", n_instances=3000)
