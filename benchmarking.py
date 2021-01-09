# ann models
import torch
from tensorflow import keras
import ML_Model.ANN.model as model
import ML_Model.ANN_TF.model_ann as model_tf

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


def get_distances(data, factual, counterfactual):
    """
    Computes distances 1 to 4
    :param data: Dataframe with original data
    :param factual: List of features
    :param counterfactual: List of features
    :return: Array of distances 1 to 4
    """
    d1 = measure.distance_d1(factual, counterfactual)
    d2 = measure.distance_d2(factual, counterfactual, data)
    d3 = measure.distance_d3(factual, counterfactual, data)
    d4 = measure.distance_d4(factual, counterfactual)

    return np.array([d1, d2, d3, d4])


def get_cost(data, factual, counterfactual):
    """
    Compute cost function
    :param data: Dataframe with original data
    :param factual: List of features
    :param counterfactual: List of features
    :return: Array of cost 1 and 2
    """
    bin_edges, norm_cdf = measure.compute_cdf(data.values)

    cost1 = measure.cost_1(factual, counterfactual, norm_cdf, bin_edges)
    cost2 = measure.cost_2(factual, counterfactual, norm_cdf, bin_edges)

    return np.array([cost1, cost2])


def compute_measurements(data, test_instances, list_of_cfs, continuous_features, target_name, model,
                         normalized=False):
    """
    Compute all measurements together and print them on the console
    :param data: Dataframe of whole data
    :param test_instances: List of Dataframes of original instance
    :param list_of_cfs: List of Dataframes of counterfactual instance
    :param continuous_features: List with continuous features
    :param target_name: String with column name of label
    :param model: ML model we want to analyze
    :param normalized: Boolean, indicates if test_instances and counterfactuals are already normalized
    :return:
    """  #

    N = len(test_instances)
    distances = np.zeros((4, 1))
    costs = np.zeros((2, 1))
    redundancy = 0

    # Columns of Dataframe for later usage
    columns = data.columns.values
    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)

    # normalize original data
    norm_data = preprocessing.normalize(data, target_name)

    for i in range(N):
        test_instance = test_instances[i]
        counterfactuals = list_of_cfs[i]  # Each list entry could be a Dataframe with more than 1 entry

        # Normalize factual and counterfactual to normalize measurements
        if not normalized:
            test_instance = preprocessing.normalize_instance(data, test_instance, continuous_features).values.tolist()[0]
            counterfactuals = preprocessing.normalize_instance(data, counterfactuals, continuous_features).values.tolist()
            counterfactual = counterfactuals[0]  # First compute measurements for only one instance with one cf
        else:
            test_instance = test_instance.values.tolist()[0]
            counterfactual = counterfactuals.values.tolist()[0]

        distances_temp = get_distances(norm_data, test_instance, counterfactual).reshape((-1, 1))
        distances += distances_temp

        # Distances are ok for now
        '''
        costs_temp = get_cost(norm_data, test_instance, counterfactual).reshape((-1, 1))
        costs += costs_temp
        '''

        # Preprocessing for redundancy
        encoded_factual = preprocessing.one_hot_encode_instance(norm_data, pd.DataFrame([test_instance], columns=columns),
                                                                cat_features)
        encoded_factual = encoded_factual.drop(columns=target_name)
        encoded_counterfactual = preprocessing.one_hot_encode_instance(norm_data,
                                                                       pd.DataFrame([counterfactual], columns=columns),
                                                                       cat_features)
        encoded_counterfactual = encoded_counterfactual.drop(columns=target_name)

        redundancy += measure.redundancy(encoded_factual.values, encoded_counterfactual.values, model)

    distances *= (1 / N)
    costs *= (1 / N)
    redundancy *= (1 / N)

    print('dist(x; x^F)_1: {}'.format(distances[0]))
    print('dist(x; x^F)_2: {}'.format(distances[1]))
    print('dist(x; x^F)_3: {}'.format(distances[2]))
    print('dist(x; x^F)_4: {}'.format(distances[3]))
    print('Redundancy: {}'.format(redundancy))
    # Distancea are ok for now
    # print('cost(x^CF; x^F)_1: {}'.format(costs[0]))
    # print('cost(x^CF; x^F)_2: {}'.format(costs[1]))

    yNN = measure.yNN(list_of_cfs, data, target_name, 5, cat_features, continuous_features, model)

    print('YNN: {}'.format(yNN))
    print('==============================================================================\n')


def compute_H_minus(data, enc_norm_data,  ml_model, label):
    """
    Computes H^{-} dataset, which contains all samples that are labeled with 0 by a black box classifier f.
    :param data: Dataframe with plain unchanged data
    :param enc_norm_data: Dataframe with normalized and encoded data
    :param ml_model: Black Box Model f (ANN, SKlearn MLP)
    :param label: String, target name
    :return: Dataframe
    """

    H_minus = data.copy()

    # loose ground truth label
    enc_data = enc_norm_data.drop(label, axis=1)

    # predict labels
    if isinstance(ml_model, model.ANN):
        predictions = np.round(ml_model(torch.from_numpy(enc_data.values).float()).detach().numpy())
    elif isinstance(ml_model, model_tf.Model_Tabular):
        predictions = ml_model.model.predict(enc_data.values)
        predictions = np.argmax(predictions, axis=1)
    else:
        raise Exception('Black-Box-Model is not yet implemented')
    H_minus['predictions'] = predictions.tolist()

    # get H^-
    H_minus = H_minus.loc[H_minus['predictions'] == 0]
    H_minus = H_minus.drop(['predictions'], axis=1)

    return H_minus


def main():
    # Get DICE counterfactuals for Adult Dataset
    data_path = 'Datasets/Adult/'
    data_name = 'adult_full.csv'
    target_name = 'income'

    # Load ANN
    # model_path = 'ML_Model/Saved_Models/ANN/2020-10-29_13-13-55_input_104_lr_0.002_te_0.34.pt'
    # ann = model.ANN(104, 64, 16, 8, 1)

    model_path = 'ML_Model/Saved_Models/ANN/2020-12-13_20-43-50_input_20_lr_0.002_te_0.35.pt'
    ann = model.ANN(20, 18, 9, 3, 1)

    # Load TF ANN (for CEM)
    model_path_tf_13 = 'ML_Model/Saved_Models/ANN_TF/ann_tf_adult_full_input_13'
    ann_tf_13 = model_tf.Model_Tabular(13, 18, 9, 3, 2, restore=model_path_tf_13, session=None, use_prob=True)
    # Load TF ANN (for Action Sequence)
    model_path_tf = 'ML_Model/Saved_Models/ANN_TF/ann_tf_adult_full_input_20'
    ann_tf = model_tf.Model_Tabular(20, 18, 9, 3, 2, restore=model_path_tf, session=None, use_prob=True)
    # Load Pytorch ANN
    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Define data with original values
    data = pd.read_csv(data_path + data_name)
    columns = data.columns
    continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)

    # Process data (normalize and encode)
    norm_data = preprocessing.normalize_instance(data, data, continuous_features)
    label_data = norm_data[target_name]
    enc_data = preprocessing.robust_binarization(norm_data, cat_features, continuous_features)
    enc_data[target_name] = label_data
    oh_data = preprocessing.one_hot_encode_instance(norm_data, norm_data, cat_features)
    oh_data[target_name] = label_data

    # Instances we want to explain
    querry_instances_tf13 = compute_H_minus(data, enc_data, ann_tf_13, target_name)
    querry_instances = compute_H_minus(data, oh_data, ann, target_name)
    # querry_instances = querry_instances.head(10)  # Only for testing because of the size of querry_instances
    querry_instances = querry_instances.head(3)  # Only for testing because of the size of querry_instances

    """
        Below we can start to define counterfactual models and start benchmarking
    """
    
    '''
    # Compute CLUE counterfactuals; so far, this one requires pytorch model
    test_instances, counterfactuals, times = clue_explainer.get_counterfactual(data_path, data_name, 'adult', querry_instances, cat_features,
                                                                       continuous_features, target_name, ann)
    '''
    # Compute FACE counterfactuals
    test_instances, counterfactuals, times = face_explainer.get_counterfactual(data_path, data_name, 'adult', querry_instances, cat_features,
                                                                       continuous_features, target_name, ann_tf_13, 'knn')
    
    # Compute FACE measurements
    print('==============================================================================')
    print('Measurement results for FACE on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann)
    
    
    
    # Compute Growing Spheres counterfactuals
    test_instances, counterfactuals, times = gs_explainer.get_counterfactual(data_path, data_name, 'adult', querry_instances, cat_features,
                                                                       continuous_features, target_name, ann_tf_13)
    
    # Compute GS measurements
    print('==============================================================================')
    print('Measurement results for GS on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann)
    

    # Compute CEM counterfactuals
    ## TODO: currently AutoEncoder (AE) and ANN models have to be pretrained; automate this!
    ## TODO: as input: 'ann_tf', 'whether AE should be trained'
    test_instances, counterfactuals, times = cem_explainer.get_counterfactual(data_path, data_name, 'adult', querry_instances, cat_features,
                                                                       continuous_features, target_name, ann_tf_13)

    # Compute CEM measurements
    print('==============================================================================')
    print('Measurement results for CEM on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann)

    '''

    # Compute DICE counterfactuals
    test_instances, counterfactuals, times = dice_examples.get_counterfactual(data_path, data_name, querry_instances,
                                                                       target_name, ann, continuous_features, 1)

    # Compute DICE measurements
    print('==============================================================================')
    print('Measurement results for DICE on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann)


    # Compute DICE with VAE
    test_instances, counterfactuals, times = dice_examples.get_counterfactual_VAE(data_path, data_name, querry_instances,
                                                                            target_name, ann, continuous_features, 1,
                                                                            pretrained=1)
    
    # Compute DICE VAE measurements
    print('==============================================================================')
    print('Measurement results for DICE with VAE on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann)

  '''

    # Compute Actionable Recourse Counterfactuals
    test_instances, counterfactuals, times = ac_explainer.get_counterfactuals(data_path, data_name, 'adult', ann,
                                                                       continuous_features, target_name, False,
                                                                       querry_instances)
    # Compute AR measurements
    print('==============================================================================')
    print('Measurement results for Actionable Recourse')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann)
    # Compute Action Sequence counterfactuals
    # Declare options for Action Sequence
    options = {
        'model_name': 'adult',
        'mode': 'vanilla',
        'length': 4,
        'actions': adult_actions
    }
    test_instances, counterfactuals = act_seq_examples.get_counterfactual(data_path, data_name, querry_instances,
                                                                          target_name, ann_tf,
                                                                          continuous_features, 1,
                                                                          options, [0., 1.])

    # Compute AS measurements
    print('==============================================================================')
    print('Measurement results for Action Sequence on Adult')
    compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
                         normalized=True)


if __name__ == "__main__":
    # execute only if run as a script
    main()
