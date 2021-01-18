import pandas as pd
import CF_Models.dice_ml as dice_ml_git
import dice_ml
import timeit
import torch

import ML_Model.ANN.model as mod
import ML_Model.ANN_TF.model_ann as mod_tf
import numpy as np
import library.measure as measure
import library.data_processing as preprocessing

import tensorflow as tf

from dice_ml.utils import helpers


def get_counterfactual(dataset_path, dataset_filename, instances, target_name, model, features, number_of_cf, backend,
                       model_path=None):
    """
    Compute counterfactual for DICE
    :param dataset_path: String; Path to folder of dataset
    :param dataset_filename: String; Filename
    :param instances: Dataframe; Instances to generate counterfactuals
    :param target_name: String, Class of label
    :param model: Pytorch model
    :param features: List of continuous feature
    :param number_of_cf: Number of counterfactuals to compute
    :param backend: String, Decides to use Tensorflow or Pytorch model ['PYT' for Pytorch, 'TF1' for Tensorflow 1,
                    'TF2' for Tensorflow 2
    :param model_path: String, Path to Tensorflow model
    :return: Counterfactual object
    """
    test_instances, counterfactuals, times_list = [], [], []
    # import dataset
    path = dataset_path
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)

    # build dice model
    dice_data = dice_ml.Data(dataframe=dataset, continuous_features=features, outcome_name=target_name)
    if backend == 'TF1':
        dice_model = dice_ml.Model(model_path=model_path,
                                   backend=backend)
    elif backend == 'TF2':
        raise NotImplementedError()
    else:
        dice_model = dice_ml.Model(model=model, backend=backend)

    # initiate DICE
    exp = dice_ml.Dice(dice_data, dice_model)

    # query instance to create examples
    query_instances = instances.drop([target_name], axis=1)

    # generate counterfactuals
    for i in range(query_instances.shape[0]):
        query_instance = query_instances.iloc[i].to_dict()
        start = timeit.default_timer()
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=number_of_cf, desired_class="opposite")
        stop = timeit.default_timer()
        time_taken = stop - start

        times_list.append(time_taken)
        # Define factual and counterfactual
        test_instances.append(dice_exp.org_instance)
        # Check if CEs are really counterfactual, if not set CE as nan
        ce = np.array(dice_exp.final_cfs_sparse)
        inst = dice_exp.test_instance
        if isinstance(model, mod.ANN):
            pred_inst = np.round(
                model(torch.from_numpy(inst).float()).detach().numpy()).squeeze()
            pred_ce = np.round(
                model(torch.from_numpy(ce).float()).detach().numpy()).squeeze()
        elif isinstance(model, mod_tf.Model_Tabular):
            # TODO: Test if DICE works correct with TF
            pred_inst = model.model.predict(inst)
            pred_inst = np.argmax(pred_inst, axis=1)
            pred_ce = model.model.predict(ce)
            pred_ce = np.argmax(pred_ce, axis=1)

        if pred_inst != pred_ce:
            counterfactuals.append(dice_exp.final_cfs_df_sparse)
        else:
            ce = dice_exp.final_cfs_df_sparse
            ce[:] = np.nan
            counterfactuals.append(ce)

    counterfactuals_df = pd.DataFrame(np.array(counterfactuals).squeeze(), columns=instances.columns)
    instances_df = pd.DataFrame(np.array(test_instances).squeeze(), columns=instances.columns)

    # Success rate & drop not successful counterfactuals & process remainder
    success_rate, counterfactuals_indeces = measure.success_rate_and_indices(
        counterfactuals_df[features].astype('float64'))
    counterfactuals_df = counterfactuals_df.iloc[counterfactuals_indeces]
    instances_df = instances_df.iloc[counterfactuals_indeces]

    # Collect in list making use of pandas
    instances_list = []
    counterfactuals_list = []

    for i in range(counterfactuals_df.shape[0]):
        counterfactuals_list.append(
            pd.DataFrame(counterfactuals_df.iloc[i].values.reshape((1, -1)), columns=counterfactuals_df.columns))
        instances_list.append(pd.DataFrame(instances_df.iloc[i].values.reshape((1, -1)), columns=instances_df.columns))

    return instances_list, counterfactuals_list, times_list, success_rate


def get_counterfactual_VAE(dataset_path, dataset_filename, instances, target_name, model, features, number_of_cf,
                           pretrained, backend, model_path=None):
    """
    Compute counterfactual for DICE with VAE
    :param dataset_path: String; Path to folder of dataset
    :param instances: Dataframe; Instances to generate counterfactuals
    :param dataset_filename: String; Filename
    :param target_name: String, Class of label
    :param model: Pytorch model
    :param features: List of continuous feature
    :param number_of_cf: Number of counterfactuals to compute
    :param pretrained: int; 1 for pretrained and 0 for training
    :param backend: String, Decides to use Tensorflow or Pytorch model ['PYT' for Pytorch, 'TF1' for Tensorflow 1,
                    'TF2' for Tensorflow 2
    :param model_path: String, Path to Tensorflow model
    :return: Counterfactual object
    """
    test_instances, counterfactuals, times_list = [], [], []

    # import dataset
    path = dataset_path
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)

    # load ML model
    ann = model

    dice_data = dice_ml_git.Data(dataframe=dataset, continuous_features=features, outcome_name=target_name,
                                 test_size=0.1,
                                 data_name=file_name)

    # build dice model
    if backend == 'PYT':
        backend = {'model': 'pytorch_model.PyTorchModel',
                   'explainer': 'feasible_base_vae.FeasibleBaseVAE'}

        dice_model = dice_ml_git.Model(model=ann, backend=backend)
    elif backend == 'TF1':
        backend = {'model': 'keras_tensorflow_model.KerasTensorFlowModel',
                   'explainer': 'feasible_base_vae.FeasibleBaseVAE'}

        dice_model = dice_ml_git.Model(model_path=model_path, backend=backend)
        dice_model.load_model()
    else:
        raise NotImplementedError()

    # initiate DiCE
    exp = dice_ml_git.Dice(dice_data, dice_model, encoded_size=10, lr=1e-2, batch_size=2048, validity_reg=42.0,
                           margin=0.165, epochs=50, wm1=1e-2, wm2=1e-2, wm3=1e-2)

    exp.train(pre_trained=pretrained)

    # query instance to create examples
    query_instances = instances.drop([target_name], axis=1)

    # generate counterfactuals
    for i in range(query_instances.shape[0]):
        query_instance = query_instances.iloc[i].to_dict()
        start = timeit.default_timer()
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=number_of_cf, desired_class="opposite")
        stop = timeit.default_timer()
        time_taken = stop - start
        times_list.append(time_taken)

        a = dice_exp.org_instance
        b = dice_exp.final_cfs_df
        # Define factual and counterfactual
        test_instances.append(dice_exp.org_instance)
        counterfactuals.append(dice_exp.final_cfs_df)

    # concatenate lists of data frames
    test_instances = pd.concat(test_instances)
    counterfactuals = pd.concat(counterfactuals)

    # Success rate & drop not successful counterfactuals & process remainder
    success_rate, counterfactuals_indeces = measure.success_rate_and_indices(counterfactuals[features].astype('float64'))
    counterfactuals = counterfactuals.iloc[counterfactuals_indeces]
    test_instances = test_instances.iloc[counterfactuals_indeces]

    # Collect in list making use of pandas
    instances_list = []
    counterfactuals_list = []
    for i in range(counterfactuals.shape[0]):
        counterfactuals_list.append(
            pd.DataFrame(counterfactuals.iloc[i].values.reshape((1, -1)), columns=counterfactuals.columns))
        instances_list.append(pd.DataFrame(instances.iloc[i].values.reshape((1, -1)), columns=instances.columns))

    return instances_list, counterfactuals_list, times_list, success_rate
