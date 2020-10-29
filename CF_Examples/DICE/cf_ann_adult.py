import torch
import pickle
import pandas as pd

from torch import nn

import ML_Model.ANN.model as model
import CF_Models.dice_ml as dice_ml
import library.data_processing as processing

from CF_Models.dice_ml.utils import helpers

# import dataset
path = '../../Datasets/Adult/'
file_name = 'adult_full.csv'
dataset = pd.read_csv(path + file_name)

# get name of categorical features
with open(path + 'cat_names.txt', 'rb') as f:
    categorical_features = pickle.load(f)

# normalize continuous features
target_name = 'income'
# dataset = processing.normalize(dataset, target_name)

# load ML model
model_path = '../../ML_Model/Saved_Models/ANN/2020-10-29_13-13-55_input_104_lr_0.002_te_0.34.pt'
ann = model.ANN(104, 64, 16, 8, 1)
ann.load_state_dict(torch.load(model_path))

# build dice model
# given our data preprocessing we do not need dice to one-hot-encode our categorial features
features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
# features = list(dataset.columns)
# features.remove(target_name)
dice_data = dice_ml.Data(dataframe=dataset, continuous_features=features, outcome_name=target_name)
dice_model = dice_ml.Model(model=ann, backend='PYT')

# initiate DICE
exp = dice_ml.Dice(dice_data, dice_model)

# query instance to create examples
query_instance = dataset.iloc[22].to_dict()
del query_instance[target_name]

# generate counterfactuals
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")

# visualize the result
dice_exp.visualize_as_dataframe()

# print(query_instance)
