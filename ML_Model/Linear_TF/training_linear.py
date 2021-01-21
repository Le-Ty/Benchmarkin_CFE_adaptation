import model_linear
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import library.data_processing as preprocessing


def __main__():
    # choose data set
    # = 'C:/Users/fred0/Desktop/Benchmarkin_Counterfactual_Examples-main/'

    # ADULT PATH
    data_path = 'Datasets/Adult/'
    data_name = 'adult_full.csv'
    target_name = 'income'

    # COMPAS PATH
    # data_path = '../../Datasets/COMPAS/'
    # data_name = 'compas-scores.csv'
    # target_name = 'is_recid'

    one_hot = True

    data = pd.read_csv(data_path + data_name)
    columns = data.columns
    continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
    # continuous_features = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count',
    #                        'days_b_screening_arrest', 'c_days_from_compas', 'c_charge_degree', 'r_charge_degree',
    #                        'v_decile_score', 'decile_score.1', 'c_jail_time', 'r_jail_time']

    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)

    if one_hot:
        data = preprocessing.one_hot_encode_instance(data, data, cat_features)
        y = data[target_name]
        data = data.drop(columns=[target_name])
    else:
        data = pd.get_dummies(data, columns=cat_features, drop_first=True)
        y = data[target_name]
        data = data.drop(columns=[target_name])

    data_normalized = preprocessing.normalize_instance(data, data, continuous_features)

    xtrain, xtest, ytrain, ytest = train_test_split(data_normalized.values, y.values, train_size=0.7)

    # training and saving ann model
    dim_input = xtrain.shape[1]
    data_name = data_name.split('.')[0]
    model = model_linear.Train_Linear(dim_input, 2, data_name)

    model.Build_Train_Save_Model(xtrain, ytrain, xtest, ytest)


if __name__ == '__main__':
    __main__()
