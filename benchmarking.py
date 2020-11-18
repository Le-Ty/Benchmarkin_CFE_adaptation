import torch

import ML_Model.ANN.model as model
import CF_Examples.DICE.cf_ann_adult as dice_examples
import library.measure as measure
import library.data_processing as preprocessing
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def get_distances(data, factual, counterfactual):
    """
    Computes distances 1 to 4
    :param data: Dataframe with original data
    :param factual: List of features
    :param counterfactual: List of features
    :return: Tuple of distances 1 to 4
    """
    d1 = measure.distance_d1(factual, counterfactual)
    d2 = measure.distance_d2(factual, counterfactual, data)
    d3 = measure.distance_d3(factual, counterfactual, data)
    d4 = measure.distance_d4(factual, counterfactual)

    print('dist(x; x^F)_1: {}'.format(d1))
    print('dist(x; x^F)_2: {}'.format(d2))
    print('dist(x; x^F)_3: {}'.format(d3))
    print('dist(x; x^F)_4: {}'.format(d4))

    return d1, d2, d3, d4


def get_cost(data, factual, counterfactual):
    """
    Compute cost function
    :param data: Dataframe with original data
    :param factual: List of features
    :param counterfactual: List of features
    :return: Tuple of cost 1 and 2
    """
    bin_edges, norm_cdf = measure.compute_cdf(data.values)

    cost1 = measure.cost_1(factual, counterfactual, norm_cdf, bin_edges)
    cost2 = measure.cost_2(factual, counterfactual, norm_cdf, bin_edges)

    print('cost(x^CF; x^F)_1: {}'.format(cost1))
    print('cost(x^CF; x^F)_2: {}'.format(cost2))

    return cost1, cost2


def compute_measurements(data, test_instance, counterfactuals, continuous_features, target_name, model):
    # Normalize data for yNN
    data_norm = preprocessing.normalize_instance(data, data, continuous_features)
    # Columns of Dataframe for later usage
    columns = data.columns.values

    # Normalize factual and counterfactual to normalize measurements
    test_instance = preprocessing.normalize_instance(data, test_instance, continuous_features).values.tolist()[0]
    counterfactuals = preprocessing.normalize_instance(data, counterfactuals, continuous_features).values.tolist()
    counterfactual = counterfactuals[0]

    d1, d2, d3, d4 = get_distances(data, test_instance, counterfactual)
    cost1, cost2 = get_cost(data, test_instance, counterfactual)

    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)
    encoded_factual = preprocessing.one_hot_encode_instance(data, pd.DataFrame([test_instance], columns=columns),
                                                            cat_features)
    encoded_factual = encoded_factual.drop(columns=target_name)
    encoded_counterfactual = preprocessing.one_hot_encode_instance(data,
                                                                   pd.DataFrame([counterfactual], columns=columns),
                                                                   cat_features)
    encoded_counterfactual = encoded_counterfactual.drop(columns=target_name)

    redundancy = measure.redundancy(encoded_factual.values, encoded_counterfactual.values, model)

    print('Redundancy: {}'.format(redundancy))

    yNN = measure.yNN(counterfactuals, data_norm, target_name, 5, cat_features, model)

    print('YNN: {}'.format(yNN))
    print('==============================================================================\n')


def main():
    # Get DICE counterfactuals for Adult Dataset
    data_path = 'Datasets/Adult/'
    data_name = 'adult_full.csv'
    target_name = 'income'
    continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']

    # Load ANN
    model_path = 'ML_Model/Saved_Models/ANN/2020-10-29_13-13-55_input_104_lr_0.002_te_0.34.pt'
    ann = model.ANN(104, 64, 16, 8, 1)
    ann.load_state_dict(torch.load(model_path))

    # Compute DICE counterfactuals
    dice_adult_cf = dice_examples.get_counterfactual(data_path, data_name, target_name, ann, continuous_features, 1)

    # Define data with original values and normalized values
    data = dice_adult_cf.data_interface.data_df

    # Define factual and counterfactual
    test_instance = dice_adult_cf.org_instance
    counterfactuals = dice_adult_cf.final_cfs_df_sparse

    # Compute measurements
    print('Measurement results for DICE and Adult')
    compute_measurements(data, test_instance, counterfactuals, continuous_features, target_name, ann)

    # DICE with VAE
    dice_adult_cf_vae = dice_examples.get_counterfactual_VAE(data_path, data_name, target_name, ann,
                                                             continuous_features, 1)

    # get Counterfactual
    test_instance = dice_adult_cf_vae.org_instance
    counterfactuals = dice_adult_cf_vae.final_cfs_df

    # Compute measurements
    print('Measurement results for DICE with VAE and Adult')
    compute_measurements(data, test_instance, counterfactuals, continuous_features, target_name, ann)


if __name__ == "__main__":
    # execute only if run as a script
    main()
