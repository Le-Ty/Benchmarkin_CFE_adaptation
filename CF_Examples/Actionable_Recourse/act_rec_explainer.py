import lime
import lime.lime_tabular
import torch

import pandas as pd
import numpy as np
import library.data_processing as processing
import matplotlib.pyplot as plt
import ML_Model.ANN.model as mod
import ML_Model.ANN_TF.model_ann as mod_tf

from recourse.builder import RecourseBuilder
from recourse.builder import ActionSet
import timeit
from mip import cbc


def prepare_adult(data, continuous_features, label):
    """
    Prepare data according to https://github.com/ustunb/actionable-recourse/blob/master/examples/ex_01_quickstart.ipynb
    Non binary categorical data is not yet supported for actionable recourse
    :param data: Dataframe with original data
    :param continuous_features: List, with continuous features
    :param label: String, with label name
    :return: Dataframe
    """
    # Get categorical features
    columns = data.columns
    cat_features = processing.get_categorical_features(columns, continuous_features, label)

    # Actionable Recourse is yet not able to support non binary categorical data
    # therefore we will only keep 'sex' and dismiss every other categorical feature
    adult_df = data.copy()
    processing.categorize_binary(adult_df, 'sex', 'Female')
    cat_features.remove('sex')
    adult_df = adult_df.drop(cat_features, axis=1)

    return adult_df


def prepare_adult_tf13(data, continuous_features, label):
    """
    Prepare data according to https://github.com/ustunb/actionable-recourse/blob/master/examples/ex_01_quickstart.ipynb
    Non binary categorical data is not yet supported for actionable recourse
    :param data: Dataframe with original data
    :param continuous_features: List, with continuous features
    :param label: String, with label name
    :return: Dataframe
    """
    # Get categorical features
    columns = data.columns
    cat_features = processing.get_categorical_features(columns, continuous_features, label)

    # Actionable Recourse is yet not able to support non binary categorical data
    adult_df13 = data.copy()
    label_data = adult_df13[label]
    adult_df13 = processing.robust_binarization(adult_df13, cat_features, continuous_features)
    adult_df13[label] = label_data
    return adult_df13


def build_lime(data, cat_feat, cont_feat, label, dataset_name):
    """
    Define a LIME explainer on dataset
    :param data: Dataframe with original data
    :param cat_feat: List with categorical data
    :param cont_feat: List with continuous data
    :param label: String with label name
    :param dataset_name: String, Given the dataset_name we decide which preprocessing we use
    :return: LimeExplainer
    """
    # Data preparation
    y = data[label]
    X = data.drop(label, axis=1)

    if dataset_name == 'adult':
        X = processing.one_hot_encode_instance(X, X, cat_feat)
    elif dataset_name == 'adult_tf13':
        X = processing.robust_binarization(X, cat_feat, cont_feat)
    X = processing.normalize(X)

    lime_exp = lime.lime_tabular.LimeTabularExplainer(training_data=X.values,
                                                      training_labels=y,
                                                      feature_names=X.columns,
                                                      discretize_continuous=False,
                                                      categorical_names=cat_feat)

    return lime_exp


def get_lime_coefficients(data, lime_expl, model, instance, categorical_features, continuous_features, label,
                          dataset_name):
    """
    Actionable Recourse is not implemented for non-linear models and non binary categorical data.
    To mitigate the second issue, we have to use LIME to compute coefficients for our Black Box Model.
    :param data: Dataframe with original data
    :param lime_expl: LimeExplainer
    :param model: Black Box Model
    :param instance: Dataframe
    :param categorical_features: List with categorical data
    :param continuous_features: List with continuous data
    :param label: String with label name
    :param dataset_name: String, Given the dataset_name we decide which preprocessing we use
    :return: List of LIME-Explanations
    """
    # Prepare instance
    if dataset_name == 'adult':
        inst_to_expl = processing.one_hot_encode_instance(data, instance, categorical_features)
    elif dataset_name == 'adult_tf13':
        inst_to_expl = pd.DataFrame(instance.values.reshape((1, -1)), columns=instance.index.values)

    inst_to_expl = processing.normalize_instance(data, inst_to_expl, continuous_features)
    inst_to_expl = inst_to_expl.drop(label, axis=1)

    # Make model prediction an convert it for lime to prob class prediction
    if isinstance(model, mod.ANN):
        explanations = lime_expl.explain_instance(np.squeeze(inst_to_expl.values), model.prob_predict)
    elif isinstance(model, mod_tf.Model_Tabular):
        explanations = lime_expl.explain_instance(np.squeeze(inst_to_expl.values), model.model.predict)
    else:
        raise Exception('Model not yet implemented')

    return explanations.as_list()


def get_counterfactuals(dataset_path, dataset_filename, dataset_name, model, continuous_features, label, is_linear,
                        instances):
    test_instances, counterfactuals = [], []

    # model_path = 'ML_Model/Saved_Models/ANN/2020-12-13_20-43-50_input_20_lr_0.002_te_0.35.pt'
    # model = model.ANN(20, 18, 9, 3, 1)

    # Load TF ANN (for CEM)
    # model_path_tf_13 = 'ML_Model/Saved_Models/ANN_TF/ann_tf_adult_full_input_13'
    # model = model_tf.Model_Tabular(13, 18, 9, 3, 2, restore=model_path_tf_13, session=None, use_prob=True)
    # Load TF ANN (for Action Sequence)
    # model_path_tf = 'ML_Model/Saved_Models/ANN_TF/ann_tf_adult_full_input_20'
    # ann_tf = model_tf.Model_Tabular(20, 18, 9, 3, 2, restore=model_path_tf, session=None, use_prob=True)
    # Load Pytorch ANN
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # import dataset
    path = dataset_path
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)
    categorical_features = processing.get_categorical_features(dataset.columns, continuous_features, label)

    # select the correct data preparation
    if dataset_name == 'adult':
        prep_data = prepare_adult(dataset, continuous_features, label)
    elif dataset_name == 'adult_tf13':
        prep_data = prepare_adult_tf13(dataset, continuous_features, label)
    else:
        print('Not yet implemented')
        prep_data = dataset

    # set up training sets for actionable recourse
    y = prep_data[label]
    X = prep_data.drop(label, axis=1)
    action_set = ActionSet(X=X)
    # set immutable features
    action_set['age'].mutable = False
    action_set['age'].actionable = False
    action_set['sex'].mutable = False
    action_set['sex'].actionable = False

    # Actionable recourse is only defined on linear models
    # To use more complex models, they propose to use local approximation models like LIME

    if not is_linear:
        times_list = []
        lime_explainer = build_lime(dataset, categorical_features, continuous_features, label, dataset_name)
        # create coefficients for each instance
        if dataset_name == 'adult_tf13':
            label_data = instances[label]
            instances = processing.robust_binarization(instances, categorical_features, continuous_features)
            instances[label] = label_data

        for i in range(instances.shape[0]):
            instance = instances.iloc[i]
            start = timeit.default_timer()
            top_10_coeff = get_lime_coefficients(dataset, lime_explainer, model, instance, categorical_features,
                                                 continuous_features, label, dataset_name)
            # Match LIME Coefficients with actionable recourse data
            # if LIME coef. is in ac_columns then use coefficient else 0
            ac_columns = X.columns
            rest_columns = [x for x in instances.columns if x not in ac_columns]
            coefficients = np.zeros(ac_columns.shape)
            for i, feature in enumerate(ac_columns):
                for t in top_10_coeff:
                    if t[0].find(feature) != -1:
                        coefficients[i] += t[1]

            # Align action set to coefficients
            action_set.set_alignment(coefficients=coefficients)

            # Build counterfactuals
            rest_df = instance[rest_columns].values.reshape((1, -1))
            rest_df = pd.DataFrame(rest_df, columns=rest_columns)
            inst_for_ac = instance[ac_columns].values.reshape((1, -1))
            inst_for_ac = pd.DataFrame(inst_for_ac, columns=ac_columns)
            if dataset_name == 'adult':
                inst_for_ac.loc[inst_for_ac['sex'] == 'Female', 'sex'] = 1
                inst_for_ac.loc[inst_for_ac['sex'] == 'Male', 'sex'] = 0
            rb = RecourseBuilder(
                optimizer='cbc',
                coefficients=coefficients,
                action_set=action_set,
                x=inst_for_ac.values
            )

            # Fit AC and build counterfactual
            ac_fit = rb.fit()
            actions = ac_fit['actions']
            counterfactual = inst_for_ac.values + actions
            counterfactual = pd.DataFrame(counterfactual, columns=ac_columns)
            counterfactual[rest_columns] = rest_df[rest_columns]
            if dataset_name == 'adult':
                counterfactual.loc[counterfactual['sex'] == 1, 'sex'] = 'Female'
                counterfactual.loc[counterfactual['sex'] == 0, 'sex'] = 'Male'
            counterfactual = counterfactual[
                instances.columns]  # Arrange instance and counterfactual in same column order

            # Record time
            stop = timeit.default_timer()
            time_taken = stop - start
            times_list.append(time_taken)

            # Prepare counterfactual for prediction
            # y_test = counterfactual['income']  # For test to compare label of original and counterfactual
            if isinstance(model, mod.ANN):
                counterfactual_pred = processing.one_hot_encode_instance(dataset, counterfactual, categorical_features)
                counterfactual_pred = processing.normalize_instance(dataset, counterfactual_pred, continuous_features)
                counterfactual_pred = counterfactual_pred.drop(label, axis=1)

                # Test output of counterfactual
                groundtruth = instance[label]
                prediction = np.round(
                    model(torch.from_numpy(counterfactual_pred.values[0]).float()).detach().numpy()).squeeze()
            elif isinstance(model, mod_tf.Model_Tabular):
                counterfactual_pred = processing.normalize_instance(dataset, counterfactual, continuous_features)
                counterfactual_pred = counterfactual_pred.drop(label, axis=1)

                # Test output of counterfactual
                groundtruth = instance[label]
                prediction = model.model.predict(counterfactual_pred.values)
                prediction = np.argmax(prediction, axis=1)
            else:
                raise Exception('Model not yet implemented')

            counterfactual['income'] = prediction

            if (groundtruth != prediction):
                counterfactuals.append(counterfactual)
                instance = pd.DataFrame(instance.values.reshape((1, -1)), columns=instances.columns)
                print(instance)
                test_instances.append(instance)

    else:
        start = timeit.default_timer()
        raise Exception('Linear models are not yet implemented')
        stop = timeit.default_timer()
        time_taken = stop - start

    print(test_instances)
    print(counterfactuals)

    return test_instances, counterfactuals, times_list
