import pandas as pd
import CF_Models.dice_ml as dice_ml_git
import dice_ml
import timeit

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
        counterfactuals.append(dice_exp.final_cfs_df_sparse)

    return test_instances, counterfactuals, times_list


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

    dice_data = dice_ml_git.Data(dataframe=dataset, continuous_features=features, outcome_name=target_name, test_size=0.1,
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

        # Define factual and counterfactual
        test_instances.append(dice_exp.org_instance)
        counterfactuals.append(dice_exp.final_cfs_df)

    return test_instances, counterfactuals, times_list
