import pickle
import pandas as pd

import CF_Models.dice_ml as dice_ml


def get_counterfactual(dataset_path, dataset_filename, target_name, model, features, number_of_cf):
    """
    Compute counterfactual for DICE
    :param dataset_path: String; Path to folder of dataset
    :param dataset_filename: String; Filename
    :param target_name: String, Class of label
    :param model: Pytorch model
    :param features: List of continuous feature
    :param number_of_cf: Number of counterfactuals to compute
    :return: Counterfactual object
    """
    # import dataset
    # path = '../../Datasets/Adult/'
    path = dataset_path
    # file_name = 'adult_full.csv'
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)

    # get name of categorical features
    with open(path + 'cat_names.txt', 'rb') as f:
        categorical_features = pickle.load(f)

    # normalize continuous features
    # target_name = 'income'
    # dataset = processing.normalize(dataset, target_name)

    # load ML model
    # model_path = '../../ML_Model/Saved_Models/ANN/2020-10-29_13-13-55_input_104_lr_0.002_te_0.34.pt'
    # ann = model.ANN(104, 64, 16, 8, 1)
    # ann.load_state_dict(torch.load(model_path))
    ann = model

    # build dice model
    # given our data preprocessing we do not need dice to one-hot-encode our categorial features
    # features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
    features = features
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
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=number_of_cf, desired_class="opposite")

    # visualize the result
    # dice_exp.visualize_as_dataframe()

    return dice_exp


def get_counterfactual_VAE(dataset_path, dataset_filename, target_name, model, features, number_of_cf):
    """
    Compute counterfactual for DICE with VAE
    :param dataset_path: String; Path to folder of dataset
    :param dataset_filename: String; Filename
    :param target_name: String, Class of label
    :param model: Pytorch model
    :param features: List of continuous feature
    :param number_of_cf: Number of counterfactuals to compute
    :return: Counterfactual object
    """
    # import dataset
    # path = '../../Datasets/Adult/'
    path = dataset_path
    # file_name = 'adult_full.csv'
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)

    # load ML model
    ann = model

    # build dice model
    backend = {'model': 'pytorch_model.PyTorchModel',
               'explainer': 'feasible_base_vae.FeasibleBaseVAE'}

    dice_data = dice_ml.Data(dataframe=dataset, continuous_features=features, outcome_name=target_name, test_size=0.1,
                             data_name='adult')
    dice_model = dice_ml.Model(model=ann, backend=backend)

    # initiate DiCE
    exp = dice_ml.Dice(dice_data, dice_model, encoded_size=10, lr=1e-2, batch_size=2048, validity_reg=42.0,
                       margin=0.165, epochs=5, wm1=1e-2, wm2=1e-2, wm3=1e-2)

    exp.train(pre_trained=0)

    # query instance to create examples
    query_instance = dataset.iloc[22].to_dict()
    del query_instance[target_name]

    # generate counterfactuals
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=number_of_cf, desired_class="opposite")

    # dice_exp.visualize_as_dataframe()

    return dice_exp
