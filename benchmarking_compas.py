# ann models
import torch
import tensorflow as tf
import pandas as pd
# from tensorflow import Graph, Session
import ML_Model.ANN.model as model
import ML_Model.ANN_TF.model_ann as model_tf
from benchmarking import compute_measurements, compute_H_minus

# CE models
import CF_Examples.DICE.dice_explainer as dice_examples
import CF_Examples.Actionable_Recourse.act_rec_explainer as ac_explainer
import CF_Examples.CEM.cem_explainer as cem_explainer
import CF_Examples.Growing_Spheres.gs_explainer as gs_explainer
import CF_Examples.FACE.face_explainer as face_explainer
import CF_Examples.CLUE.clue_explainer as clue_explainer
import CF_Examples.Action_Sequence.action_sequence_explainer as act_seq_examples
from CF_Examples.Action_Sequence.adult_actions import actions as adult_actions
from sklearn.neural_network import MLPClassifier

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

    # Load ANNs
    model_path = 'ML_Model/Saved_Models/ANN/2021-01-17_08-58-41_compas_input_20_lr_0.0001_te_0.42.pt'
    ann = model.ANN(20, 18, 9, 3, 1)
    # Load TF ANN
    model_path_tf_17 = 'ML_Model/Saved_Models/ANN_TF/ann_tf_compas-scores_input_17'
    ann_tf_17 = model_tf.Model_Tabular(17, 18, 9, 3, 2, restore=model_path_tf_17, session=None, use_prob=True)
    # Load TF ANN
    model_path_tf = 'ML_Model/Saved_Models/ANN_TF/ann_tf_compas-scores_input_20'
    ann_tf = model_tf.Model_Tabular(20, 18, 9, 3, 2, restore=model_path_tf, session=None, use_prob=True)
    # Load Pytorch ANN
    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Define data with original values
    data = pd.read_csv(data_path + data_name)
    columns = data.columns
    # assign updated column names
    continuous_features = ['age', 'juv-fel-count', 'decile-score', 'juv-misd-count', 'juv-other-count', 'priors-count',
                           'days-b-screening-arrest', 'c-days-from-compas', 'c-charge-degree', 'r-charge-degree',
                           'v-decile-score', 'decile-score-01', 'c-jail-time', 'r-jail-time']
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
    querry_instances_tf17 = compute_H_minus(data, enc_data, ann_tf_17, target_name)
    querry_instances_tf = compute_H_minus(data, oh_data, ann_tf, target_name)
    querry_instances = compute_H_minus(data, oh_data, ann, target_name)
    querry_instances = querry_instances.head(10)  # Only for testing because of the size of querry_instances
    querry_instances_tf17 = querry_instances_tf17.head(5)
    querry_instances_tf = querry_instances_tf.head(3)

    """
        Below we can start to define counterfactual models and start benchmarking
    """
    
    
    '''
    # Compute CLUE counterfactuals; This one requires the pytorch model
    test_instances, counterfactuals, times, success_rate = clue_explainer.get_counterfactual(data_path, data_name,
                                                                                             'compas', querry_instances,
                                                                                             cat_features,
                                                                                             continuous_features,
                                                                                             target_name, ann,
                                                                                             train_vae=False)

    # Compute CLUE measurements
    print('==============================================================================')
    print('Measurement results for CLUE on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                         immutable, normalized=True)

    '''


    # Compute FACE counterfactuals
    test_instances, counterfactuals, times, success_rate = face_explainer.get_counterfactual(data_path, data_name,
                                                                                             'compas',
                                                                                             querry_instances_tf17,
                                                                                             cat_features,
                                                                                             continuous_features,
                                                                                             target_name, ann_tf_17,
                                                                                             'knn')

    # Compute FACE measurements
    print('==============================================================================')
    print('Measurement results for FACE on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_17,
                         immutable, normalized=True, one_hot=False)

    # Compute GS counterfactuals
    test_instances, counterfactuals, times, success_rate = gs_explainer.get_counterfactual(data_path, data_name,
                                                                                           'compas',
                                                                                           querry_instances_tf17,
                                                                                           cat_features,
                                                                                           continuous_features,
                                                                                           target_name, ann_tf_17)

    # Compute GS measurements
    print('==============================================================================')
    print('Measurement results for GS on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_17,
                         immutable, normalized=True, one_hot=False)


if __name__ == "__main__":
    # execute only if run as a script
    main()
