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


# CE models
# import CF_Examples.DICE.dice_explainer as dice_explainer
# import CF_Examples.Actionable_Recourse.act_rec_explainer as ac_explainer
# import CF_Examples.CEM.cem_explainer as cem_explainer
# import CF_Examples.Growing_Spheres.gs_explainer as gs_explainer
# import CF_Examples.FACE.face_explainer as face_explainer
# import CF_Examples.CLUE.clue_explainer as clue_explainer
# import CF_Examples.Action_Sequence.action_sequence_explainer as act_seq_examples
# from CF_Examples.Action_Sequence.compas_actions import actions as compas_actions
# from sklearn.neural_network import MLPClassifier
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
    ann_tf_16 = load_model("/home/uni/TresoritDrive/XY/uni/WS2021/BA/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/outputs/models/Compas/ANN_predictor.h5")




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
        querry_instances_tf16.to_csv("CF_Input/Compas/ANN/query_instances.csv",index = False)

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

        # print(counterfactuals)
        # print(test_instances)



        #TODO give own data cuz of predictor
        df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_16,
                                immutable, times, success_rate, normalized=True, one_hot=False)

        df_results.to_csv('Results/Compas/{}/{}.csv'.format(classifier_name, model_name))




    # # Compute CLUE counterfactuals; This one requires the pytorch model
    # test_instances, counterfactuals, times, success_rate = clue_explainer.get_counterfactual(data_path, data_name,
    #                                                                                          'compas', querry_instances,
    #                                                                                          cat_features,
    #                                                                                          continuous_features,
    #                                                                                          target_name, ann,
    #                                                                                          train_vae=True)
    #
    # # Compute CLUE measurements
    # print('==============================================================================')
    # print('Measurement results for CLUE on COMPAS')
    # compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
    #                      immutable, times, success_rate, normalized=True)
    #
    # #
    # # Compute FACE counterfactuals
    # with graph1.as_default():
    #     with ann_16_sess.as_default():
    #         test_instances, counterfactuals, times, success_rate = face_explainer.get_counterfactual(data_path, data_name,
    #                                                                                          'compas',
    #                                                                                          querry_instances_tf16,
    #                                                                                          cat_features,
    #                                                                                          continuous_features,
    #                                                                                          target_name, ann_tf_16,
    #                                                                                          'knn')
    #
    #         # Compute FACE measurements
    #         print('==============================================================================')
    #         print('Measurement results for FACE on COMPAS')
    #         compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_16,
    #                      immutable, times, success_rate, normalized=True, one_hot=False)
    #
    # # Compute GS counterfactuals
    # with graph1.as_default():
    #     with ann_16_sess.as_default():
    #         test_instances, counterfactuals, times, success_rate = gs_explainer.get_counterfactual(data_path, data_name,
    #                                                                                        'compas',
    #                                                                                        querry_instances_tf16,
    #                                                                                        cat_features,
    #                                                                                        continuous_features,
    #                                                                                        target_name, ann_tf_16)
    #
    #         # Compute GS measurements
    #         print('==============================================================================')
    #         print('Measurement results for GS on COMPAS')
    #         compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_16,
    #                      immutable, times, success_rate, normalized=True, one_hot=False)
    #
    #
    # with graph1.as_default():
    #     with ann_16_sess.as_default():
    #         test_instances, counterfactuals, times, success_rate = cem_explainer.get_counterfactual(data_path,
    #                                                                                                 data_name,
    #                                                                                                 'adult',
    #                                                                                                 querry_instances_tf16,
    #                                                                                                 cat_features,
    #                                                                                                 continuous_features,
    #                                                                                                 target_name,
    #                                                                                                 ann_tf_16,
    #                                                                                                 ann_16_sess)
    #
    #         # Compute CEM measurements
    #         print('==============================================================================')
    #         print('Measurement results for CEM on Adult')
    #         compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_16,
    #                              immutable, times, success_rate, normalized=True, one_hot=False)
    #
    #
    # # Compute DICE counterfactuals
    # test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual(data_path, data_name,
    #                                                                                          querry_instances,
    #                                                                                          target_name, ann,
    #                                                                                          continuous_features,
    #                                                                                          1, 'PYT')
    #
    # # Compute DICE measurements
    # print('==============================================================================')
    # print('Measurement results for DICE on COMPAS')
    # compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
    #                      immutable, times, success_rate, normalized=False, one_hot=True)
    #
    # '''
    #
    # # Compute DICE with VAE
    # test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual_VAE(data_path, data_name,
    #                                                                                              querry_instances,
    #                                                                                              target_name, ann,
    #                                                                                              continuous_features,
    #                                                                                              1, pretrained=0,
    #                                                                                              backend='PYT')
    #
    # # Compute DICE VAE measurements
    # print('==============================================================================')
    # print('Measurement results for DICE with VAE on Adult')
    # compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
    #                      immutable, normalized=False, one_hot=True)
    #
    # # Compute Actionable Recourse Counterfactuals
    # with graph1.as_default():
    #     with ann_16_sess.as_default():
    #         test_instances, counterfactuals, times, success_rate = ac_explainer.get_counterfactuals(data_path,
    #                                                                                                 data_name,
    #                                                                                                 'compas',
    #                                                                                                 ann_tf_16,
    #                                                                                                 continuous_features,
    #                                                                                                 target_name, False,
    #                                                                                                 querry_instances_tf16)
    #
    #         # Compute AR measurements
    #         print('==============================================================================')
    #         print('Measurement results for Actionable Recourse')
    #         compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_16,
    #                              immutable, times, success_rate, normalized=False, one_hot=False, encoded=True)
    #
    #
    # # Declare options for Action Sequence
    # with graph2.as_default():
    #     with ann_19_sess.as_default():
    #         options = {
    #             'model_name': 'compas',
    #             'mode': 'vanilla',
    #             'length': 4,
    #             'actions': compas_actions
    #         }
    #         test_instances, counterfactuals, times, success_rate = act_seq_examples.get_counterfactual(data_path,
    #                                                                                                    data_name,
    #                                                                                                    querry_instances_tf,
    #                                                                                                    target_name,
    #                                                                                                    ann_tf,
    #                                                                                                    ann_19_sess,
    #                                                                                                    continuous_features,
    #                                                                                                    options,
    #                                                                                                    [0., 1.])
    #
    #         # Compute AS measurements
    #         print('==============================================================================')
    #         print('Measurement results for Action Sequence on Adult')
    #         compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
    #                              immutable, times, success_rate, normalized=True, one_hot=True, separator=':')


    # path_cfe = '/home/uni/TresoritDrive/XY/uni/WS2021/BA/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/out_for_ben/Compas/ANN/'
    # model_name = "di_cfe"
    #
    # file = open(path_cfe + "counterfactuals.pickle",'rb')
    # counterfactuals = pickle.load(file)
    # file.close()
    #
    # file = open(path_cfe + "test_instances.pickle",'rb')
    # test_instances = pickle.load(file)
    # file.close()
    #
    # file = open(path_cfe + "times_list.pickle",'rb')
    # times = pickle.load(file)
    # file.close()
    #
    # file = open(path_cfe + "success_rate.pickle",'rb')
    # success_rate = pickle.load(file)
    # file.close()
    #
    # df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf_13,
    #                         immutable, times, success_rate, normalized=True, one_hot=False)
    #
    # df_results.to_csv('Results/Compas/{}/{}.csv'.format(classifier_name, model_name))
    #




if __name__ == "__main__":
    # execute only if run as a script
    main()
