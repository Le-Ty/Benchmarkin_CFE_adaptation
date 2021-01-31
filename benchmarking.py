# ann models
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# from tensorflow import Graph, Session
# import ML_Model.ANN.model as model
# import ML_Model.ANN_TF.model_ann as model_tf

# # CE models
# import CF_Examples.DICE.dice_explainer as dice_explainer
# import CF_Examples.Actionable_Recourse.act_rec_explainer as ac_explainer
# import CF_Examples.CEM.cem_explainer as cem_explainer
# import CF_Examples.Growing_Spheres.gs_explainer as gs_explainer
# import CF_Examples.FACE.face_explainer as face_explainer
# import CF_Examples.CLUE.clue_explainer as clue_explainer
# import CF_Examples.Action_Sequence.action_sequence_explainer as act_seq_examples
# from CF_Examples.Action_Sequence.adult_actions import actions as adult_actions
# from sklearn.neural_network import MLPClassifier
import CF_Examples.counterfact_expl.CE.experiments.run_synthetic as ce_explainer

# others
import library.measure as measure
import library.data_processing as preprocessing
import pandas as pd
import numpy as np
import random


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


def compute_measurements(data, test_instances, list_of_cfs, continuous_features, target_name, model, immutable,
                         times, success_rate, normalized=False, one_hot=True, encoded=False, separator='_'):
    """
    Compute all measurements together and print them on the console
    :param data: Dataframe of whole data
    :param test_instances: List of Dataframes of original instance
    :param list_of_cfs: List of Dataframes of counterfactual instance
    :param continuous_features: List with continuous features
    :param target_name: String with column name of label
    :param model: ML model we want to analyze
    :param normalized: Boolean, indicates if test_instances and counterfactuals are already normalized
    :param one_hot: Boolean, indicates whether test_instances & counterfactual instances should become one-hot or binarized
    :param encoded: Boolean, indicates whether test_instances & counterfactual are already encoded
    :param immutable: List; list of immutable features
    :param separator: String, determines the separator for one-hot-encoding
    :return:
    """  #

    N = len(test_instances)
    if N == 0:
        print('No counterfactuals found!')
        return
    distances = np.zeros((4, 1))
    costs = np.zeros((2, 1))
    redundancy = 0
    violation = 0

    # Columns of Dataframe for later usage
    columns = data.columns.values
    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)

    # normalize original data
    norm_data = preprocessing.normalize(data, target_name)

    # Initialize lists
    distances_list = []
    redundancy_list = []
    violation_list = []

    for i in range(N):
        test_instance = test_instances[i]
        counterfactuals = list_of_cfs[i]  # Each list entry could be a Dataframe with more than 1 entry

        # Normalize factual and counterfactual to normalize measurements
        if not normalized:
            test_instance = preprocessing.normalize_instance(data, test_instance, continuous_features).values.tolist()[
                0]
            counterfactuals = preprocessing.normalize_instance(data, counterfactuals,
                                                               continuous_features).values.tolist()
            counterfactual = counterfactuals[0]  # First compute measurements for only one instance with one cf
        else:
            test_instance = test_instance.values.tolist()[0]
            counterfactual = counterfactuals.values.tolist()[0]

        distances_temp = get_distances(norm_data, test_instance, counterfactual).reshape((-1, 1))
        distances += distances_temp
        distances_list.append(np.array(distances_temp).reshape(1, -1))

        # Distances are ok for now
        '''
        costs_temp = get_cost(norm_data, test_instance, counterfactual).reshape((-1, 1))
        costs += costs_temp
        '''

        # Preprocessing for redundancy & yNN
        if not encoded:
            if one_hot:
                encoded_factual = preprocessing.one_hot_encode_instance(norm_data,
                                                                        pd.DataFrame([test_instance], columns=columns),
                                                                        cat_features, separator=separator)
                encoded_factual = encoded_factual.drop(columns=target_name)
                encoded_counterfactual = preprocessing.one_hot_encode_instance(norm_data,
                                                                               pd.DataFrame([counterfactual],
                                                                                            columns=columns),
                                                                               cat_features, separator=separator)
                encoded_counterfactual = encoded_counterfactual.drop(columns=target_name)
            else:
                data_no_target = data.drop(columns=target_name)
                columns = data_no_target.columns.tolist()
                # print(data_no_target.head(10))
                # print(test_instances)

                encoded_factual = pd.DataFrame([test_instance], columns=columns + [target_name])
                encoded_factual = encoded_factual.drop(columns=target_name)
                # print(encoded_factual)
                encoded_factual = preprocessing.robust_binarization_2(encoded_factual, data_no_target,
                                                                      cat_features, continuous_features)

                # print(encoded_factual)
                encoded_counterfactual = pd.DataFrame([counterfactual], columns=columns + [target_name])
                encoded_counterfactual = encoded_counterfactual.drop(columns=target_name)
                encoded_counterfactual = preprocessing.robust_binarization_2(encoded_counterfactual, data_no_target,
                                                                             cat_features, continuous_features)
        else:
            encoded_factual = pd.DataFrame([test_instance], columns=columns)
            encoded_factual = encoded_factual.drop(columns=target_name)
            encoded_counterfactual = pd.DataFrame([counterfactual], columns=columns)
            encoded_counterfactual = encoded_counterfactual.drop(columns=target_name)

        redundancy_measure = measure.redundancy(encoded_factual.values, encoded_counterfactual.values, model)
        redundancy_list.append(redundancy_measure)
        redundancy += redundancy_measure

        if len(immutable) > 0:
            violation_measure = measure.constraint_violation(encoded_counterfactual, encoded_factual, immutable,
                                                             separator)
            violation_list.append(violation_measure)
            violation += violation_measure

    distances *= (1 / N)
    costs *= (1 / N)
    redundancy *= (1 / N)
    if len(immutable) > 0:
        violation *= (1 / len(immutable) * N)
    else:
        violation_list, violation = 0, 0

    print('dist(x; x^F)_1: {}'.format(distances[0]))
    print('dist(x; x^F)_2: {}'.format(distances[1]))
    print('dist(x; x^F)_3: {}'.format(distances[2]))
    print('dist(x; x^F)_4: {}'.format(distances[3]))
    # Distances are ok for now
    # print('cost(x^CF; x^F)_1: {}'.format(costs[0]))
    # print('cost(x^CF; x^F)_2: {}'.format(costs[1]))

    yNN = measure.yNN(list_of_cfs, data, target_name, 5,
                      cat_features, continuous_features,
                      model, one_hot, normalized, encoded)
    avg_time = np.mean(np.array(times))

    print('Redundancy: {}'.format(redundancy))
    print('Constraint Violation: {}'.format(violation))
    print('YNN: {}'.format(yNN))
    print('Success Rate: {}'.format(success_rate))
    print('Average Time: {}'.format(avg_time))
    print('==============================================================================\n')

    # Convert results to data frames
    df_distances = pd.DataFrame(np.array(distances_list)[:, 0, :], columns=['ell0', 'ell1', 'ell2', 'ell-inf'])
    df_violation = pd.DataFrame(np.array(violation_list).reshape(-1, 1), columns=['violation'])
    df_redundancy = pd.DataFrame(np.array(redundancy_list).reshape(-1, 1), columns=['redundancy'])
    df_ynn = pd.DataFrame(np.array(yNN).reshape(-1, 1), columns=['ynn'])
    df_success = pd.DataFrame(np.array(success_rate).reshape(-1, 1), columns=['success'])
    df_time = pd.DataFrame(np.array(avg_time).reshape(-1, 1), columns=['avgtime'])

    df_results = pd.concat([df_distances, df_redundancy, df_violation, df_ynn, df_success, df_time], axis=1)

    return df_results


def compute_H_minus(data, enc_norm_data, ml_model, label):
    """
    Computes H^{-} dataset: contains all samples that are predicted as 0 (negative class) by black box classifier f.
    :param data: Dataframe with plain unchanged data
    :param enc_norm_data: Dataframe with normalized and encoded data
    :param ml_model: Black Box Model f (ANN, SKlearn MLP)
    :param label: String, target name
    :return: Dataframe
    """ #

    H_minus = data.copy()

    # loose ground truth label
    enc_data = enc_norm_data.drop(label, axis=1)

    # predict labels
    # if isinstance(ml_model, model.ANN):
    #     predictions = np.round(ml_model(torch.from_numpy(enc_data.values).float()).detach().numpy()).squeeze()
    # elif isinstance(ml_model, model_tf.Model_Tabular):
    predictions = ml_model.predict(enc_data.values)
    with pd.option_context('display.max_rows', 20, 'display.max_columns', 20):
        print(enc_data)


    #which index has the max
    predictions = np.argmax(predictions, axis=1)

    # else:
    #     raise Exception('Black-Box-Model is not yet implemented')
    H_minus['predictions'] = predictions.tolist()

    # get H^-
    H_minus["predictions"] = H_minus["predictions"].astype(int)
    H_minus = H_minus.loc[H_minus["predictions"] == 0]
    H_minus = H_minus.drop(["predictions"], axis=1)

    return H_minus


def main():

    # Get DICE counterfactuals for Adult Dataset
    data_path = 'Datasets/Adult/'
    data_name = 'adult_full.csv'
    target_name = 'income'
    classifier_name = "ANN"
    save = False
    benchmark = True

    #Load model
    ann_tf_13 = load_model("/home/uni/TresoritDrive/XY/uni/WS2021/BA/ablation/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/outputs/models/Adult/" + classifier_name + "_predictor.h5")


    # Define data with original values
    data = pd.read_csv(data_path + data_name)
    columns = data.columns
    continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
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
    querry_instances_tf_13 = compute_H_minus(data, enc_data, ann_tf_13, target_name)
    querry_instances_tf_13 = querry_instances_tf_13.head(100)

    if save:
        querry_instances_tf_13.to_csv("CF_Input/Adult/"+ classifier_name + "/query_instances.csv",index = False)



    """
        Below we can start to define counterfactual models and start benchmarking
    """

    '''
    # Compute CLUE counterfactuals; This one requires the pytorch model
    model_name = 'clue'
    test_instances, counterfactuals, times, success_rate = clue_explainer.get_counterfactual(data_path, data_name,
                                                                                             'adult', querry_instances,
                                                                                             cat_features,
                                                                                             continuous_features,
                                                                                             target_name, ann,
                                                                                             train_vae=False)

    # Compute CLUE measurements
    print('==============================================================================')
    print('Measurement results for CLUE on Adult')
    df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                         immutable, times, success_rate, normalized=True, one_hot=True)
    #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))


    # Compute FACE counterfactuals
    model_name = 'face'
    with graph1.as_default():
        with ann_13_sess.as_default():
            test_instances, counterfactuals, times, success_rate = face_explainer.get_counterfactual(data_path, data_name,
                                                                                                'adult',
                                                                                                querry_instances_tf13,
                                                                                                cat_features,
                                                                                                continuous_features,
                                                                                                target_name, ann_tf_13,
                                                                                                'knn')

            # Compute FACE measurements
            print('==============================================================================')
            print('Measurement results for FACE on Adult')
            df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_13,
                            immutable, times, success_rate, normalized=True, one_hot=False)
            #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))


    # Compute GS counterfactuals
    model_name = 'gs'
    with graph1.as_default():
        with ann_13_sess.as_default():
            test_instances, counterfactuals, times, success_rate = gs_explainer.get_counterfactual(data_path, data_name,
                                                                                            'adult',
                                                                                            querry_instances_tf13,
                                                                                            cat_features,
                                                                                            continuous_features,
                                                                                            target_name, ann_tf_13)

        # Compute GS measurements
            print('==============================================================================')
            print('Measurement results for GS on Adult')
            df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_13,
                            immutable, times, success_rate, normalized=True, one_hot=False)
            #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))



    # Compute CEM counterfactuals
    model_name = 'cem'
    ## TODO: as input: 'whether AE should be trained'
    with graph1.as_default():
        with ann_13_sess.as_default():
            test_instances, counterfactuals, times, success_rate = cem_explainer.get_counterfactual(data_path, data_name,
                                                                                                'adult',
                                                                                                querry_instances_tf13,
                                                                                                cat_features,
                                                                                                continuous_features,
                                                                                                target_name,
                                                                                                ann_tf_13,
                                                                                                ann_13_sess)

        # Compute CEM measurements
            print('==============================================================================')
            print('Measurement results for CEM on Adult')
            df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_13,
                            immutable, times, success_rate, normalized=True, one_hot=False)
            #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))


    # Compute DICE counterfactuals
    model_name = 'dice'
    test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual(data_path, data_name,
                                                                                             querry_instances,
                                                                                             target_name, ann,
                                                                                             continuous_features,
                                                                                             1,
                                                                                             'PYT')

    # THIS MODEL DOES NOT WORK YET! WEIRD 'GRAPH NOT FOUND ISSUE'
    # test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual(data_path, data_name,
    #                                                                           querry_instances,
    #                                                                           target_name, ann_tf, continuous_features,
    #                                                                           1, 'TF1', model_path_tf)

    # Compute DICE measurements
    print('==============================================================================')
    print('Measurement results for DICE on Adult')
    df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                         immutable, times, success_rate, one_hot=True)
    #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))

    '''
    # # Compute DICE with VAE
    # model_name = 'dice_vae'
    # test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual_VAE(data_path, data_name,
    #                                                                                querry_instances,
    #                                                                                target_name, ann,
    #                                                                                continuous_features,
    #                                                                                1, pretrained=1,
    #                                                                                backend='PYT')
    #
    # # Compute DICE VAE measurements
    # print('==============================================================================')
    # print('Measurement results for DICE with VAE on Adult')
    # df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
    #                      immutable, times, success_rate, one_hot=True)
    # #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))
    #
    #
    # # Compute Actionable Recourse Counterfactuals
    # model_name = 'ar'
    # with graph1.as_default():
    #     with ann_13_sess.as_default():
    #         test_instances, counterfactuals, times, success_rate = ac_explainer.get_counterfactuals(data_path, data_name,
    #                                                                                             'adult_tf13',
    #                                                                                             ann_tf_13,
    #                                                                                             continuous_features,
    #                                                                                             target_name, False,
    #                                                                                             querry_instances_tf13)
    #         # Compute AR measurements
    #         print('==============================================================================')
    #         print('Measurement results for Actionable Recourse')
    #         df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_13,
    #                         immutable, times, success_rate, normalized=False, one_hot=False, encoded=True)
    #         #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))


    # Compute Action Sequence counterfactuals
    # model_name = 'as'
    # with graph2.as_default():
    #     with ann_20_sess.as_default():
    #
    #         # Declare options for Action Sequence
    #         options = {
    #             'model_name': 'adult',
    #             'mode': 'vanilla',
    #             'length': 4,
    #             'actions': adult_actions
    #         }
    #         test_instances, counterfactuals, times, success_rate = act_seq_examples.get_counterfactual(data_path, data_name,
    #                                                                                                querry_instances_tf,
    #                                                                                                target_name,
    #                                                                                                ann_tf,
    #                                                                                                ann_20_sess,
    #                                                                                                continuous_features,
    #                                                                                                options,
    #                                                                                                [0., 1.])
    #
    #         # Compute AS measurements
    #         print('==============================================================================')
    #         print('Measurement results for Action Sequence on Adult')
    #         df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
    #                         immutable, times, success_rate, normalized=True, one_hot=True)
    #         #df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))
    #


    # im_feata = ["age", "sex_Male"]
    # #
    # # # Compute Actionable Recourse Counterfactuals
    # test_instances, counterfactuals, times, success_rate = ce_explainer.run_synthetic(query = querry_instances_tf_13,
    #                                                         im_feat = im_feata,
    #                                                         train_steps = 10000,
    #                                                         model = "ANN", dataset = "Adult", number_cf = 20, train_AAE = False)

    if benchmark:
        path_cfe = '/home/uni/TresoritDrive/XY/uni/WS2021/BA/ablation/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/out_for_ben/Adult/' + classifier_name + "/"
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

        # print(counterfactuals)
        # print(test_instances)



        #TODO give own data cuz of predictor
        df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_13,
                                immutable, times, success_rate, normalized=True, one_hot=False)

        df_results.to_csv('Results/Adult/{}/{}.csv'.format(classifier_name, model_name))




if __name__ == "__main__":
    # execute only if run as a script
    main()
