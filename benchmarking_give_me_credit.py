# ann models
import torch
import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import keras
import numpy as np

# from tensorflow import Session, Graph
import ML_Model.ANN.model as model
import ML_Model.ANN_TF.model_ann as model_tf
from benchmarking import compute_measurements, compute_H_minus

# Linear models
import ML_Model.Linear.model as model_lin
import ML_Model.Linear_TF.model_linear as model_tf_lin

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



def main():
    # Get COMPAS Dataset
    data_path = 'Datasets/Give_Me_Some_Credit/'
    data_name = 'give_me_processed.csv'
    target_name = 'SeriousDlqin2yrs'

    # Define data with original values
    data = pd.read_csv(data_path + data_name)
    classifier_name = 'Linear'
    save = False
    benchmark = True

    count = 0
    df = np.array(data['SeriousDlqin2yrs'])
    print(df)
    for i in range(len(df)):
        if df[i] == 0:
            count +=1

    print(count)


    '''
         Loading ANNs
    '''
    # Load ANN
    def weighted_binary_cross_entropy(t_true, y_pred):
        loss = 0.7 * (t_true * tf.math.log(y_pred)) + \
                0.3 * ((1 - t_true) * tf.math.log(1 - y_pred))
        return tf.math.negative(tf.reduce_mean(loss, axis=-1))

    ann_tf = load_model("/home/uni/TresoritDrive/XY/uni/WS2021/BA/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/outputs/models/GMC/"+ classifier_name +"_predictor.h5", compile = False)
    ann_tf.compile(
        optimizer='rmsprop',  # works better than sgd
        loss= weighted_binary_cross_entropy,
        metrics=['accuracy'])



    columns = data.columns
    continuous_features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                       'NumberRealEstateLoansOrLines' ,'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

    immutable = ['age', 'NumberOfDependents']
    cat_features = preprocessing.get_categorical_features(columns, continuous_features, target_name)

    # Process data (normalize and encode)
    norm_data = preprocessing.normalize_instance(data, data, continuous_features)
    label_data = norm_data[target_name]
    enc_data = preprocessing.robust_binarization(norm_data, cat_features, continuous_features)
    enc_data[target_name] = label_data
    oh_data = preprocessing.one_hot_encode_instance(norm_data, norm_data, cat_features)
    oh_data[target_name] = label_data

    '''
        Querry instances for ANN
    '''
    # Instances we want to explain
    querry_instances_tf = compute_H_minus(data, enc_data, ann_tf, target_name)
    print(len(querry_instances_tf))
    querry_instances_tf = querry_instances_tf.head(100)

    if save:
        querry_instances_tf.to_csv("CF_Input/GMC/"+ classifier_name +"/query_instances.csv",index = False)

    print(len(querry_instances_tf))

    '''
        Querry instances for linear model
    '''
    # Instances we want to explain
    # with graph2.as_default():
    #     with lin_sess.as_default():
    #         querry_instances_tf_lin = compute_H_minus(data, enc_data, lin_tf, target_name)
    #         querry_instances_tf_lin = querry_instances_tf_lin.head(10)

    # querry_instances_lin = compute_H_minus(data, oh_data, lin, target_name)
    # querry_instances_lin = querry_instances_lin.head(10)  # Only for testing because of the size of querry_instances

    """
        Below we can start to define counterfactual models and start benchmarking
    """

    '''
        Benchmarking for ANN model
    '''



    if benchmark:
        path_cfe = '/home/uni/TresoritDrive/XY/uni/WS2021/BA/Benchmarkin_Counterfactual_Examples/CF_Examples/counterfact_expl/CE/out_for_ben/GMC/' + classifier_name + "/"
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


        #TODO give own data cuz of predictor
        df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
                                immutable, times, success_rate, normalized=True, one_hot=False)

        df_results.to_csv('Results/GMC/{}/{}.csv'.format(classifier_name, model_name))














    '''
    # Compute CLUE counterfactuals; This one requires the pytorch model
    model_name = 'clue'
    test_instances, counterfactuals, times, success_rate = clue_explainer.get_counterfactual(data_path,
                                                                                             data_name,
                                                                                             'give-me',
                                                                                             querry_instances,
                                                                                             cat_features,
                                                                                             continuous_features,
                                                                                             target_name,
                                                                                             ann,
                                                                                             train_vae=False)


    # Compute CLUE measurements
    print('==============================================================================')
    print('Measurement results for CLUE on Adult')
    df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                         immutable, times, success_rate, normalized=True, one_hot=True)
    #df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))

    #
    # '''
    # # Compute FACE counterfactuals
    # model_name = 'face-eps'
    # with graph1.as_default():
    #     with ann_sess.as_default():
    #         test_instances, counterfactuals, times, success_rate = face_explainer.get_counterfactual(data_path,
    #                                                                                                  data_name,
    #                                                                                                  'give-me',
    #                                                                                                   querry_instances_tf,
    #                                                                                                   cat_features,
    #                                                                                                   continuous_features,
    #                                                                                                   target_name,
    #                                                                                                   ann_tf,
    #                                                                                                   'epsilon')
    #         # Compute FACE-EPS measurements
    #         print('==============================================================================')
    #         print('Measurement results for FACE-EPS on Adult')
    #         df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
    #                         immutable, times, success_rate, normalized=True, one_hot=False)
    #         #df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))
    #
    #
    # model_name = 'face-knn'
    # with graph1.as_default():
    #     with ann_sess.as_default():
    #         test_instances, counterfactuals, times, success_rate = face_explainer.get_counterfactual(data_path,
    #                                                                                                  data_name,
    #                                                                                                  'give-me',
    #                                                                                                   querry_instances_tf,
    #                                                                                                   cat_features,
    #                                                                                                   continuous_features,
    #                                                                                                   target_name,
    #                                                                                                   ann_tf,
    #                                                                                                   'knn')
    #
    #         # Compute FACE measurements
    #         print('==============================================================================')
    #         print('Measurement results for FACE-knn on Adult')
    #         df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
    #                         immutable, times, success_rate, normalized=True, one_hot=False)
    #         #df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))


    '''
    # Compute GS counterfactuals
    model_name = 'gs'
    with graph1.as_default():
        with ann_sess.as_default():
            test_instances, counterfactuals, times, success_rate = gs_explainer.get_counterfactual(data_path,
                                                                                            data_name,
                                                                                            'give-me',
                                                                                            querry_instances_tf,
                                                                                            cat_features,
                                                                                            continuous_features,
                                                                                            target_name,
                                                                                            ann_tf)

        # Compute GS measurements
            print('==============================================================================')
            print('Measurement results for GS on Adult')
            df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
                            immutable, times, success_rate, normalized=True, one_hot=False)
            #df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))



    # Compute CEM counterfactuals
    model_name = 'cem'
    ## TODO: as input: 'whether AE should be trained'
    with graph1.as_default():
        with ann_sess.as_default():
            test_instances, counterfactuals, times, success_rate = cem_explainer.get_counterfactual(data_path, data_name,
                                                                                                'give-me',
                                                                                                querry_instances_tf,
                                                                                                cat_features,
                                                                                                continuous_features,
                                                                                                target_name,
                                                                                                ann_tf,
                                                                                                ann_sess,
                                                                                                kappa=0.1, beta=0.9, gamma=0.0)

        # Compute CEM measurements
            print('==============================================================================')
            print('Measurement results for CEM on Adult')
            df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann_tf,
                            immutable, times, success_rate, normalized=True, one_hot=False)
            #df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))


        model_name = 'cem-vae'
        ## TODO: as input: 'whether AE should be trained'
        with graph1.as_default():
            with ann_sess.as_default():
                test_instances, counterfactuals, times, success_rate = cem_explainer.get_counterfactual(data_path,
                                                                                                        data_name,
                                                                                                        'give-me',
                                                                                                        querry_instances_tf,
                                                                                                        cat_features,
                                                                                                        continuous_features,
                                                                                                        target_name,
                                                                                                        ann_tf,
                                                                                                        ann_sess,
                                                                                                        kappa=0.1, beta=0.0, gamma=6.0)

                # Compute CEM measurements
                print('==============================================================================')
                print('Measurement results for CEM VAE on Adult')
                df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features,
                                                  target_name, ann_tf,
                                                  immutable, times, success_rate, normalized=True, one_hot=False)
                #df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))




    # Compute DICE counterfactuals
    model_name = 'dice'
    test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual(data_path, data_name,
                                                                                             querry_instances,
                                                                                             target_name, ann,
                                                                                             continuous_features,
                                                                                             1,
                                                                                             'PYT')

    # Compute DICE measurements
    print('==============================================================================')
    print('Measurement results for DICE on Adult')
    df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                                      immutable, times, success_rate, one_hot=True)
    # df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))


    # Compute DICE with VAE
    model_name = 'dice_vae'
    test_instances, counterfactuals, times, success_rate = dice_explainer.get_counterfactual_VAE(data_path, data_name,
                                                                                                 querry_instances,
                                                                                                 target_name, ann,
                                                                                                 continuous_features,
                                                                                                 1, pretrained=1,
                                                                                                 backend='PYT')

    # Compute DICE VAE measurements
    print('==============================================================================')
    print('Measurement results for DICE with VAE on Adult')
    df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name, ann,
                                      immutable, times, success_rate, one_hot=True)
    # df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))



    # Compute Actionable Recourse Counterfactuals
    model_name = 'ar'
    with graph1.as_default():
        with ann_sess.as_default():
            test_instances, counterfactuals, times, success_rate = ac_explainer.get_counterfactuals(data_path,
                                                                                                    data_name,
                                                                                                    'give-me',
                                                                                                    ann_tf,
                                                                                                    continuous_features,
                                                                                                    target_name,
                                                                                                    False,
                                                                                                    querry_instances_tf)

            # Compute AR measurements
            print('==============================================================================')
            print('Measurement results for Actionable Recourse')
            df_results = compute_measurements(data, test_instances, counterfactuals, continuous_features, target_name,
                                              ann_tf,
                                              immutable, times, success_rate, normalized=False, one_hot=False,
                                              encoded=True)
            # df_results.to_csv('Results/Give_Me_Some_Credit/{}/{}.csv'.format(classifier_name, model_name))


    '''
if __name__ == "__main__":
    # execute only if run as a script
    main()
