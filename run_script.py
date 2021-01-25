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

#read in query
query_instances_a = pd.read_csv("CF_Input/Adult/ANN/query_instances.csv", index_col = False)
# query_instances_a = pd.read_csv("CF_Input/Adult/Linear/query_instances.csv", index_col = False)

# query_instances_c = pd.read_csv("CF_Input/Adult/query_instances.csv", index_col = False).head(10)
#Adult
im_feata = ["age", "sex"]

# im_featg = ["NumberOfDependents"]

#COMPAS
im_featc = ["age", "sex"]

keras.backend.clear_session()


run_synthetic(im_feat = im_feata, query = query_instances_a, train_steps = 10000, model = "ANN", dataset = "Adult", number_cf = 100, train_AAE = True)
# find_ce()

#pred = sum_predictor()

#pred.predict()

# plot_shap_vals(shapvals_filepath=path + "outputs/synthetic_test/shap_values.csv",
# 	data_filepath=path + "data/synthetic/sum_synthetic.csv",
# 	indirect_output_filepath=path + "outputs/synthetic_test/indirect_influence_distributions.png",
# 	direct_output_filepath=path + "outputs/synthetic_test/direct_influence_distributions.png",
# 	predictor_filepath=path + "experiments/synthetic/models/sum_predictor.h5", n_instances=3000)
