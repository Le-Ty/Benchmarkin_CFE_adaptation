# ann models
import torch
import tensorflow as tf
import pandas as pd
import pickle
# from tensorflow import Session, Graph
import ML_Model.ANN.model as model
# import ML_Model.ANN_TF.model_ann as model_tf
from benchmarking import compute_measurements, compute_H_minus
from tensorflow.keras.models import load_model

import CF_Examples.counterfact_expl.CE.experiments.run_synthetic as ce_explainer


# others
import library.measure as measure
import library.data_processing as preprocessing
import pandas as pd
import numpy as np


def main():
    # Get COMPAS Dataset
    data_path = 'Datasets/COMPAS/'
    data_name = 'compas-scores.csv'
    target_name = 'is-recid'

    classifier_name = "ANN"
    save = False
    benchmark = True

    #linear
    # ann_tf_13 = load_model("/home/uni/TresoritDrive/XY/uni/WS2021/BA/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/outputs/models/Adult/Linear_predictor.h5")
    #ann
    ann_tf_16 = load_model("/home/uni/TresoritDrive/XY/uni/WS2021/BA/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/outputs/models/Compas/"+ classifier_name+"_predictor.h5")

    # Define data with original values
    data = pd.read_csv(data_path + data_name)
    columns = data.columns
    continuous_features = ['age', 'juv-fel-count', 'decile-score', 'juv-misd-count', 'juv-other-count', 'priors-count',
                           'days-b-screening-arrest', 'c-days-from-compas', 'c-charge-degree', 'r-charge-degree',
                           'v-decile-score', 'c-jail-time', 'r-jail-time']
    immutable = ['age', 'sex']
    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)

    # Process data (normalize and encode)
    norm_data = preprocessing.normalize_instance(data, data, continuous_features)
    label_data = norm_data[target_name]
    enc_data = preprocessing.robust_binarization(norm_data, cat_features, continuous_features)
    enc_data[target_name] = label_data
    oh_data = preprocessing.one_hot_encode_instance(norm_data, norm_data, cat_features)
    oh_data[target_name] = label_data

    # Instances we want to explain

    querry_instances_tf16 = compute_H_minus(data, enc_data, ann_tf_16, target_name)
    querry_instances_tf16 = querry_instances_tf16.head(100)
    print(querry_instances_tf16.head(10))
    print(data)

    if save:
        querry_instances_tf16.to_csv("CF_Input/Compas/" + classifier_name +"/query_instances.csv",index = False)

    """
        Below we can start to define counterfactual models and start benchmarking
    """



    if benchmark:
        path_cfe = '/home/uni/TresoritDrive/XY/uni/WS2021/BA/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/out_for_ben/Compas/' + classifier_name + "/"
        model_name = "dicfe"

        file = open(path_cfe + "counterfactuals.pickle",'rb')
        counterfactuals = pickle.load(file)
        file.close()

        file = open(path_cfe + "test_instances.pickle",'rb')
        test_instances = pickle.load(file)
        file.close()

        file = open(path_cfe + "times_list.pickle",'rb')
        times = pickle.load(file)
        file.close()

        file = open(path_cfe + "success_rate.pickle",'rb')
        success_rate = pickle.load(file)
        file.close()

        file = open(path_cfe + "direct_change.pickle",'rb')
        direct_change = pickle.load(file)
        file.close()


        #TODO give own data cuz of predictor
        df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_16,
                                immutable, times, success_rate, normalized=True, one_hot=False)

        df_direct = compute_measurements(data, test_instances, direct_change, continuous_features, target_name, ann_tf_16,
                                immutable, times, success_rate, normalized=True, one_hot=False)

        df_indirect = compute_measurements(data, direct_change, counterfactuals, continuous_features, target_name, ann_tf_16,
                                immutable, times, success_rate, normalized=True, one_hot=False)

        df_results.to_csv('Results/Compas/{}/{}.csv'.format(classifier_name, model_name))

        df_direct.to_csv('Results/Compas/{}/{}-dir.csv'.format(classifier_name, model_name))

        df_indirect.to_csv('Results/Compas/{}/{}-indir.csv'.format(classifier_name, model_name))





if __name__ == "__main__":
    # execute only if run as a script
    main()
