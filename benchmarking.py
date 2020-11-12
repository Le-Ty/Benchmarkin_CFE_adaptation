import torch

import ML_Model.ANN.model as model
import CF_Examples.DICE.cf_ann_adult as dice_examples
import library.measure as measure
import library.data_processing as preprocessing
import pandas as pd


def main():
    # Get DICE counterfactuals for Adult Dataset
    data_path = 'Datasets/Adult/'
    data_name = 'adult_full.csv'
    target_name = 'income'
    continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']

    model_path = 'ML_Model/Saved_Models/ANN/2020-10-29_13-13-55_input_104_lr_0.002_te_0.34.pt'
    ann = model.ANN(104, 64, 16, 8, 1)
    ann.load_state_dict(torch.load(model_path))

    dice_adult_cf = dice_examples.get_counterfactual(data_path, data_name, target_name, ann, continuous_features, 1)

    test_instance = dice_adult_cf.org_instance.values.tolist()[0]
    counterfactuals = dice_adult_cf.final_cfs_list_sparse
    counterfactual = counterfactuals[0]
    data = dice_adult_cf.data_interface.data_df

    d1 = measure.distance_d1(test_instance, counterfactual)
    d2 = measure.distance_d2(test_instance, counterfactual, data)
    d3 = measure.distance_d3(test_instance, counterfactual, data)
    d4 = measure.distance_d4(test_instance, counterfactual)

    print('Measurement results for DICE and Adult')
    print('dist(x; x^F)_1: {}'.format(d1))
    print('dist(x; x^F)_2: {}'.format(d2))
    print('dist(x; x^F)_3: {}'.format(d3))
    print('dist(x; x^F)_4: {}'.format(d4))

    bin_edges, norm_cdf = measure.compute_cdf(data.values)

    cost1 = measure.cost_1(test_instance, counterfactual, norm_cdf, bin_edges)
    cost2 = measure.cost_2(test_instance, counterfactual, norm_cdf, bin_edges)

    print('cost(x^CF; x^F)_1: {}'.format(cost1))
    print('cost(x^CF; x^F)_2: {}'.format(cost2))

    columns = data.columns.values
    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)
    encoded_factual = preprocessing.one_hot_encode_instance(data, pd.DataFrame([test_instance], columns=columns),
                                                            cat_features)
    encoded_factual = encoded_factual.drop(columns=target_name)
    encoded_counterfactual = preprocessing.one_hot_encode_instance(data,
                                                                   pd.DataFrame([counterfactual], columns=columns),
                                                                   cat_features)
    encoded_counterfactual = encoded_counterfactual.drop(columns=target_name)

    redundancy = measure.redundancy(encoded_factual.values, encoded_counterfactual.values, ann)

    print('Redundancy: {}'.format(redundancy))

    yNN = measure.yNN(counterfactuals, data, target_name, 5, cat_features, ann)

    print('YNN: {}'.format(yNN))

    # DICE with VAE
    dice_adult_cf_vae = dice_examples.get_counterfactual_VAE(data_path, data_name, target_name, ann,
                                                             continuous_features, 1)

    # get Counterfactual
    test_instance = dice_adult_cf_vae.org_instance.values.tolist()[0]
    counterfactuals = dice_adult_cf_vae.final_cfs_list
    counterfactual = counterfactuals[0]

    d1 = measure.distance_d1(test_instance, counterfactual)
    d2 = measure.distance_d2(test_instance, counterfactual, data)
    d3 = measure.distance_d3(test_instance, counterfactual, data)
    d4 = measure.distance_d4(test_instance, counterfactual)

    print('Measurement results for DICE with VAE and Adult')
    print('dist(x; x^F)_1: {}'.format(d1))
    print('dist(x; x^F)_2: {}'.format(d2))
    print('dist(x; x^F)_3: {}'.format(d3))
    print('dist(x; x^F)_4: {}'.format(d4))

    cost1 = measure.cost_1(test_instance, counterfactual, norm_cdf, bin_edges)
    cost2 = measure.cost_2(test_instance, counterfactual, norm_cdf, bin_edges)

    print('cost(x^CF; x^F)_1: {}'.format(cost1))
    print('cost(x^CF; x^F)_2: {}'.format(cost2))

    encoded_factual = preprocessing.one_hot_encode_instance(data, pd.DataFrame([test_instance], columns=columns),
                                                            cat_features)
    encoded_factual = encoded_factual.drop(columns=target_name)
    encoded_counterfactual = preprocessing.one_hot_encode_instance(data,
                                                                   pd.DataFrame([counterfactual], columns=columns),
                                                                   cat_features)
    encoded_counterfactual = encoded_counterfactual.drop(columns=target_name)

    redundancy = measure.redundancy(encoded_factual.values, encoded_counterfactual.values, ann)

    print('Redundancy: {}'.format(redundancy))

    yNN = measure.yNN(counterfactuals, data, target_name, 5, cat_features, ann)

    print('YNN: {}'.format(yNN))


if __name__ == "__main__":
    # execute only if run as a script
    main()
