# ann models
import torch
import tensorflow as tf
import pandas as pd
from tensorflow import Session, Graph
import ML_Model.ANN.model as model
import ML_Model.ANN_TF.model_ann as model_tf
from benchmarking import compute_measurements, compute_H_minus

# CE models
import CF_Examples.DICE.dice_explainer as dice_explainer
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
    target_name = 'is_recid'

    # Load ANNs
    model_path = 'ML_Model/Saved_Models/ANN/2021-01-17_08-58-41_compas_input_20_lr_0.0001_te_0.42.pt'
    ann = model.ANN(20, 18, 9, 3, 1)
    
    # Load TF ANN
    graph1 = Graph()
    with graph1.as_default():
        ann_17_sess = Session()
        with ann_17_sess.as_default():
            model_path_tf_17 = 'ML_Model/Saved_Models/ANN_TF/ann_tf_compas-scores_input_17'
            ann_tf_17 = model_tf.Model_Tabular(17, 18, 9, 3, 2, restore=model_path_tf_17, use_prob=True)
        
    # Load TF ANN: One-hot encoded
    graph2 = Graph()
    with graph2.as_default():
        ann_20_sess = Session()
        with ann_20_sess.as_default():
            model_path_tf = 'ML_Model/Saved_Models/ANN_TF/ann_tf_compas-scores_input_20'
            ann_tf = model_tf.Model_Tabular(20, 18, 9, 3, 2, restore=model_path_tf, use_prob=True)
        
    # Load Pytorch ANN
    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Define data with original values
    data = pd.read_csv(data_path + data_name)
    columns = data.columns
    continuous_features = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count',
                           'days_b_screening_arrest', 'c_days_from_compas', 'c_charge_degree', 'r_charge_degree',
                           'v_decile_score', 'decile_score_2', 'c_jail_time', 'r_jail_time']
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
    with graph1.as_default():
        with ann_17_sess.as_default():
            querry_instances_tf17 = compute_H_minus(data, enc_data, ann_tf_17, target_name)
            querry_instances_tf17 = querry_instances_tf17.head(8)

    with graph2.as_default():
        with ann_20_sess.as_default():
             querry_instances_tf = compute_H_minus(data, oh_data, ann_tf, target_name)
             querry_instances_tf = querry_instances_tf.head(5)

    querry_instances = compute_H_minus(data, oh_data, ann, target_name)
    querry_instances = querry_instances.head(10)  # Only for testing because of the size of querry_instances


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
    print('Measurement results for CLUE on COMPAS')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                         immutable, normalized=True)

    '''


    # Compute FACE counterfactuals
    with graph1.as_default():
        with ann_17_sess.as_default():
            test_instances, counterfactuals, times, success_rate = face_explainer.get_counterfactual(data_path, data_name,
                                                                                             'compas',
                                                                                             querry_instances_tf17,
                                                                                             cat_features,
                                                                                             continuous_features,
                                                                                             target_name, ann_tf_17,
                                                                                             'knn')

            # Compute FACE measurements
            print('==============================================================================')
            print('Measurement results for FACE on COMPAS')
            compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_17,
                         immutable, normalized=True, one_hot=False)

    # Compute GS counterfactuals
    with graph1.as_default():
        with ann_17_sess.as_default():
            test_instances, counterfactuals, times, success_rate = gs_explainer.get_counterfactual(data_path, data_name,
                                                                                           'compas',
                                                                                           querry_instances_tf17,
                                                                                           cat_features,
                                                                                           continuous_features,
                                                                                           target_name, ann_tf_17)

            # Compute GS measurements
            print('==============================================================================')
            print('Measurement results for GS on COMPAS')
            compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_17,
                         immutable, normalized=True, one_hot=False)
            
            
    with graph1.as_default():
        with ann_17_sess.as_default():
            test_instances, counterfactuals, times, success_rate = cem_explainer.get_counterfactual(data_path, data_name,
                                                                                                'adult',
                                                                                                querry_instances_tf17,
                                                                                                cat_features,
                                                                                                continuous_features,
                                                                                                target_name,
                                                                                                ann_tf_17,
                                                                                                ann_17_sess)

        # Compute CEM measurements
            print('==============================================================================')
            print('Measurement results for CEM on Adult')
            compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_17,
                            immutable, normalized=True, one_hot=False)
            

    # Compute DICE counterfactuals
    test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual(data_path, data_name,
                                                                                             querry_instances,
                                                                                             target_name, ann,
                                                                                             continuous_features,
                                                                                             1, 'PYT')

    # Compute DICE measurements
    print('==============================================================================')
    print('Measurement results for DICE on COMPAS')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                         immutable, normalized=False, one_hot=True)

    # Compute DICE with VAE
    test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual_VAE(data_path, data_name,
                                                                                                 querry_instances,
                                                                                                 target_name, ann,
                                                                                                 continuous_features,
                                                                                                 1, pretrained=0,
                                                                                                 backend='PYT')

    # Compute DICE VAE measurements
    print('==============================================================================')
    print('Measurement results for DICE with VAE on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                         immutable, normalized=False, one_hot=True)
    
    # Compute Actionable Recourse Counterfactuals
    # with ann_17_sess.as_default():
    # TODO: LIME Coefficients for ann_tf_17 on compas are to small (1e-16). 
    test_instances, counterfactuals, times, success_rate = ac_explainer.get_counterfactuals(data_path, data_name,
                                                                                            'compas',
                                                                                            ann_tf_17,
                                                                                            continuous_features,
                                                                                            target_name, False,
                                                                                            querry_instances_tf17)

    # Compute AR measurements
    print('==============================================================================')
    print('Measurement results for Actionable Recourse')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_17,
                         immutable, normalized=False, one_hot=False, encoded=True)
    '''

    # Declare options for Action Sequence
    options = {
        'model_name': 'compas',
        'mode': 'vanilla',
        'length': 4,
        'actions': adult_actions
    }
    test_instances, counterfactuals, times, success_rate = act_seq_examples.get_counterfactual(data_path, data_name,
                                                                                               querry_instances_tf,
                                                                                               target_name,
                                                                                               ann_tf,
                                                                                               ann_20_sess,
                                                                                               continuous_features,
                                                                                               options,
                                                                                               [0., 1.])

    # Compute AS measurements
    print('==============================================================================')
    print('Measurement results for Action Sequence on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
                         immutable, normalized=True, one_hot=True)
'''

if __name__ == "__main__":
    # execute only if run as a script
    main()
