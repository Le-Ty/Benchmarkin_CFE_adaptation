import sys
import model_ae
from sklearn.model_selection import train_test_split
import pandas as pd
import library.data_processing as preprocessing

def __main__():

    # choose data set
    #prefix = 'C:/Users/fred0/Desktop/Benchmarkin_Counterfactual_Examples-main/'
    data_path = 'Datasets/Adult/'
    data_name = 'adult_full.csv'
    target_name = 'income'
    
    small_model = False
    one_hot = False
    
    if data_name == 'adult_full.csv':
    
        data = pd.read_csv(data_path + data_name)
        columns = data.columns
        continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']

        data_name = data_name.split('.')[0]

        if not small_model:
    
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
            
            # training and saving autoencoder model
            dim_input = xtrain.shape[1]
            model = model_ae.Train_AE(dim_input, 20, 10, 5, data_name)
            model.Build_Train_Save_Model(xtrain, ytrain, xtest, ytest)

        
        else:
    
            to_drop = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
            data = data.drop(columns=to_drop, axis=1)
            cat_features = ['sex']
            data_with_one_hot = preprocessing.one_hot_encode_instance(data, data, cat_features)
            y = data_with_one_hot['income']
            data_with_one_hot = data_with_one_hot.drop(columns=['income'])
            data_normalized = preprocessing.normalize_instance(data_with_one_hot, data_with_one_hot,
                                                               continuous_features)
    
            xtrain, xtest, ytrain, ytest = train_test_split(data_normalized.values, y.values, train_size=0.5)

            data_name_new = 'adult_reduced'

            # training and saving autoencoder model
            dim_input = xtrain.shape[1]
            model = model_ae.Train_AE(dim_input, 6, 4, 2, data_name_new)
            model.Build_Train_Save_Model(xtrain, ytrain, xtest, ytest)


if __name__ == '__main__':

    __main__()