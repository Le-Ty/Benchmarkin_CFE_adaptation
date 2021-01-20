import lime
import lime.lime_tabular
import torch

import pandas as pd
import numpy as np
import library.data_processing as processing
import matplotlib.pyplot as plt
import ML_Model.ANN.model as mod
import ML_Model.ANN_TF.model_ann as mod_tf
import library.measure as measure

from recourse.builder import RecourseBuilder
from recourse.flipset import Flipset
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
    elif dataset_name in ['adult_tf13', 'compas']:
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
    :return: List of LIME-Explanations, interception
    """
    # Prepare instance
    if dataset_name == 'adult':
        inst_to_expl = processing.one_hot_encode_instance(data, instance, categorical_features)
    elif dataset_name in ['adult_tf13', 'compas']:
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

    return explanations.as_list(), explanations.intercept[1]


def get_counterfactuals(dataset_path, dataset_filename, dataset_name, model, continuous_features, label, is_linear,
                        instances):
    test_instances, counterfactuals = [], []

    # import dataset
    path = dataset_path
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)
    categorical_features = processing.get_categorical_features(dataset.columns, continuous_features, label)

    # select the correct data preparation
    if dataset_name == 'adult':
        prep_data = prepare_adult(dataset, continuous_features, label)
    elif dataset_name in ['adult_tf13', 'compas']:
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
    if dataset_name == 'adult_tf13':
        action_set['race'].mutable = False
        action_set['race'].actionable = False

    # Actionable recourse is only defined on linear models
    # To use more complex models, they propose to use local approximation models like LIME

    if not is_linear:
        times_list = []
        lime_explainer = build_lime(dataset, categorical_features, continuous_features, label, dataset_name)
        # create coefficients for each instance
        if dataset_name in ['adult_tf13', 'compas']:
            label_data = instances[label]
            instances = processing.robust_binarization(instances, categorical_features, continuous_features)
            instances[label] = label_data

        for i in range(instances.shape[0]):
            instance = instances.iloc[i]
            start = timeit.default_timer()
            top_10_coeff, intercept = get_lime_coefficients(dataset, lime_explainer, model, instance,
                                                            categorical_features,
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

            fb = Flipset(
                x=inst_for_ac.values,
                action_set=action_set,
                coefficients=coefficients,
                intercept=intercept
            )

            # Fit AC and build counterfactual
            fb_set = fb.populate(enumeration_type='distinct_subsets', total_items=100, cost_type='total')
            actions_flipset = fb_set.actions
            last_object = len(actions_flipset) - 1
            for idx, action in enumerate(actions_flipset):
                # counterfactual = inst_for_ac.values + actions_flipset
                counterfactual = inst_for_ac.values + action
                counterfactual = pd.DataFrame(counterfactual, columns=ac_columns)
                counterfactual[rest_columns] = rest_df[rest_columns]
                if dataset_name == 'adult':
                    counterfactual.loc[counterfactual['sex'] == 1, 'sex'] = 'Female'
                    counterfactual.loc[counterfactual['sex'] == 0, 'sex'] = 'Male'
                counterfactual = counterfactual[
                    instances.columns]  # Arrange instance and counterfactual in same column order

                # Prepare counterfactual for prediction
                # y_test = counterfactual['income']  # For test to compare label of original and counterfactual
                if isinstance(model, mod.ANN):
                    counterfactual_pred = processing.one_hot_encode_instance(dataset, counterfactual,
                                                                             categorical_features)
                    counterfactual_pred = processing.normalize_instance(dataset, counterfactual_pred,
                                                                        continuous_features)
                    counterfactual_pred = counterfactual_pred.drop(label, axis=1)
                    inst = pd.DataFrame(instance.values.reshape((1, -1)), columns=instances.columns)
                    inst = processing.normalize_instance(dataset, inst, continuous_features)
                    inst_pred = processing.one_hot_encode_instance(dataset, inst, categorical_features)
                    inst_pred = processing.normalize_instance(dataset, inst_pred, continuous_features)
                    inst_pred = inst_pred.drop(label, axis=1)

                    # Test output of counterfactual
                    prediction_inst = np.round(
                        model(torch.from_numpy(inst_pred.values[0]).float()).detach().numpy()).squeeze()
                    prediction = np.round(
                        model(torch.from_numpy(counterfactual_pred.values[0]).float()).detach().numpy()).squeeze()
                elif isinstance(model, mod_tf.Model_Tabular):
                    counterfactual_pred = processing.normalize_instance(dataset, counterfactual, continuous_features)
                    counterfactual_pred = counterfactual_pred.drop(label, axis=1)
                    inst = pd.DataFrame(instance.values.reshape((1, -1)), columns=instances.columns)
                    inst = processing.normalize_instance(dataset, inst, continuous_features)
                    inst = inst.drop(label, axis=1)

                    # Test output of counterfactual
                    prediction_inst = model.model.predict(inst.values)
                    prediction_inst = np.argmax(prediction_inst, axis=1)

                    prediction = model.model.predict(counterfactual_pred.values)
                    prediction = np.argmax(prediction, axis=1)
                else:
                    raise Exception('Model not yet implemented')

                counterfactual['income'] = prediction
                instance = pd.DataFrame(instance.values.reshape((1, -1)), columns=instances.columns)
                test_instances.append(instance)

                if (prediction_inst != prediction):
                    counterfactuals.append(counterfactual)
                    break
                elif idx == last_object:
                    counterfactual[:] = np.nan
                    counterfactuals.append(counterfactual)

            # Record time
            stop = timeit.default_timer()
            time_taken = stop - start
            times_list.append(time_taken)

        counterfactuals_df = pd.DataFrame(np.array(counterfactuals).squeeze(), columns=instances.columns)
        instances_df = pd.DataFrame(np.array(test_instances).squeeze(), columns=instances.columns)

        # Success rate & drop not successful counterfactuals & process remainder
        success_rate, counterfactuals_indeces = measure.success_rate_and_indices(
            counterfactuals_df[continuous_features].astype('float64'))
        counterfactuals_df = counterfactuals_df.iloc[counterfactuals_indeces]
        instances_df = instances_df.iloc[counterfactuals_indeces]

        # Collect in list making use of pandas
        instances_list = []
        counterfactuals_list = []

        for i in range(counterfactuals_df.shape[0]):
            counterfactuals_list.append(
                pd.DataFrame(counterfactuals_df.iloc[i].values.reshape((1, -1)), columns=counterfactuals_df.columns))
            instances_list.append(
                pd.DataFrame(instances_df.iloc[i].values.reshape((1, -1)), columns=instances_df.columns))


    else:
        start = timeit.default_timer()
        raise Exception('Linear models are not yet implemented')
        stop = timeit.default_timer()
        time_taken = stop - start

    return instances_list, counterfactuals_list, times_list, success_rate
