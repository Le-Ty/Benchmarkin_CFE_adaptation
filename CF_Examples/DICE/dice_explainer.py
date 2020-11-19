import pandas as pd

import CF_Models.dice_ml as dice_ml


def get_counterfactual(dataset_path, dataset_filename, instances, target_name, model, features, number_of_cf):
    """
    Compute counterfactual for DICE
    :param dataset_path: String; Path to folder of dataset
    :param dataset_filename: String; Filename
    :param instances: Dataframe; Instances to generate counterfactuals
    :param target_name: String, Class of label
    :param model: Pytorch model
    :param features: List of continuous feature
    :param number_of_cf: Number of counterfactuals to compute
    :return: Counterfactual object
    """
    test_instances, counterfactuals = [], []
    # import dataset
    path = dataset_path
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)

    # build dice model
    dice_data = dice_ml.Data(dataframe=dataset, continuous_features=features, outcome_name=target_name)
    dice_model = dice_ml.Model(model=model, backend='PYT')

    # initiate DICE
    exp = dice_ml.Dice(dice_data, dice_model)

    # query instance to create examples
    query_instances = instances.drop([target_name], axis=1)

    # generate counterfactuals
    for i in range(query_instances.shape[0]):
        query_instance = query_instances.iloc[i].to_dict()
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=number_of_cf, desired_class="opposite")

        # Define factual and counterfactual
        test_instances.append(dice_exp.org_instance)
        counterfactuals.append(dice_exp.final_cfs_df_sparse)

    return test_instances, counterfactuals


def get_counterfactual_VAE(dataset_path, dataset_filename, instances, target_name, model, features, number_of_cf,
                           pretrained):
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
    :return: Counterfactual object
    """
    test_instances, counterfactuals = [], []

    # import dataset
    path = dataset_path
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)

    # load ML model
    ann = model

    # build dice model
    backend = {'model': 'pytorch_model.PyTorchModel',
               'explainer': 'feasible_base_vae.FeasibleBaseVAE'}

    dice_data = dice_ml.Data(dataframe=dataset, continuous_features=features, outcome_name=target_name, test_size=0.1,
                             data_name=file_name)
    dice_model = dice_ml.Model(model=ann, backend=backend)

    # initiate DiCE
    exp = dice_ml.Dice(dice_data, dice_model, encoded_size=10, lr=1e-2, batch_size=2048, validity_reg=42.0,
                       margin=0.165, epochs=5, wm1=1e-2, wm2=1e-2, wm3=1e-2)

    exp.train(pre_trained=pretrained)

    # query instance to create examples
    query_instances = instances.drop([target_name], axis=1)

    # generate counterfactuals
    for i in range(query_instances.shape[0]):
        query_instance = query_instances.iloc[i].to_dict()
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=number_of_cf, desired_class="opposite")

        # Define factual and counterfactual
        test_instances.append(dice_exp.org_instance)
        counterfactuals.append(dice_exp.final_cfs_df)

    return test_instances, counterfactuals
