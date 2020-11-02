import torch

import ML_Model.ANN.model as model
import CF_Examples.DICE.cf_ann_adult as dice_examples
import library.measure as measure


def main():
    # Get DICE counterfactuals for Adult Dataset
    data_path = 'Datasets/Adult/'
    data_name = 'adult_full.csv'
    target_name = 'income'
    features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']

    model_path = 'ML_Model/Saved_Models/ANN/2020-10-29_13-13-55_input_104_lr_0.002_te_0.34.pt'
    ann = model.ANN(104, 64, 16, 8, 1)
    ann.load_state_dict(torch.load(model_path))

    dice_adult_cf = dice_examples.get_counterfactual(data_path, data_name, target_name, ann, features, 1)

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

    test_bin_edges, test_norm_cdf = measure.compute_cdf(data.values)

    print('End')


if __name__ == "__main__":
    # execute only if run as a script
    main()
