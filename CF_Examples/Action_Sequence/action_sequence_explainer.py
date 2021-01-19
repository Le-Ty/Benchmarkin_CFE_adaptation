import pandas as pd
import tensorflow as tf
import numpy as np
import CF_Examples.Action_Sequence.feature_wrapper as fw
import library.data_processing as processing
import CF_Examples.Action_Sequence.model_wrapper as model_wrapper
import ML_Model.ANN_TF.model_ann as model_tf
import library.measure as measure
import timeit

from CF_Examples.Action_Sequence.data_wrapper import Data_wrapper
from CF_Models.act_seq.heuristics.loader import load_heuristics
from CF_Models.act_seq.recourse.search import SequenceSearch, ParamsSearch
from CF_Models.act_seq.recourse.config import base_config
from CF_Models.act_seq.recourse.utils import get_instance_info


def get_counterfactual(dataset_path, dataset_filename, instances, target_name, model, sess, cont_features,
                       options, target_prediction):
    """
    Compute counterfactual for Action Sequence
    The structure for the dictionary option is as follows:
    options = {
        target-model: String, name of the data set [german, adult,  fanniemae, quickdraw]
        mode: String, name of the heuristic we want to use [vanilla, step-full, abs_grad, quickdraw]
        length: int, length of sequences that are used
        actions: List of Action. An Action-Object from action.py
    }
    :param dataset_path: String; Path to folder of dataset
    :param dataset_filename: String; Filename
    :param instances: Dataframe; Instances to generate counterfactuals
    :param target_name: String, Class of label
    :param model: Pytorch model
    :param cont_features: List of continuous feature
    :param options: Dictionary with parameters for Action Sequence
    :param target_prediction: List of wanted prediction, e.g. for two classes [0., 1.]
    :return: Counterfactual object
    """  #

    test_instances, counterfactuals, times_list = [], [], []
    # import dataset
    path = dataset_path
    file_name = dataset_filename
    dataset = pd.read_csv(path + file_name)

    cat_features = processing.get_categorical_features(dataset.columns, cont_features, target_name)

    # Preparing Instances
    instances_oh = processing.one_hot_encode_instance(dataset, instances, cat_features)
    instances_oh = processing.normalize_instance(dataset, instances_oh, cont_features)
    instances_oh = instances_oh.drop(target_name, axis=1)

    model_wr = model_wrapper.Model_wrapper(model)
    # for layer in model_wr.model.model.layers:
    #     if hasattr(layer, 'kernel_initializer'):
    #         layer.kernel.initializer.run(session=session)

    # Load data with wrapper for Action Sequence
    data = Data_wrapper(dataset, target_name, cat_features, cont_features)

    # Build correct feature order between dataset and data
    ordered_columns = data.get_feature_order() + [target_name]  # Action Sequence needs target label to be at last
    # Loose one-hot-encoded column names for ordering dataset
    last = ''
    ordered_columns_temp = []
    for col in ordered_columns:
        f = col.split('_')[0]
        if last != f:
            ordered_columns_temp.append(f)
            last = f
    ordered_columns = ordered_columns_temp
    old_ordered_columns = dataset.columns
    dataset = dataset[ordered_columns]

    # Classification label in Action sequence is encoded in Low and High
    dataset[target_name].loc[dataset[target_name] == 0] = 'Low'
    dataset[target_name].loc[dataset[target_name] == 1] = 'High'
    raw_features, mapping = fw.create_feature_mapping(dataset, target_name)
    features = fw.loader(raw_features)

    # Choose correct actions for method
    actions = [action_cls(features) for action_cls in options['actions']]

    for name, feature in features.items():
        feature.initialize_tf_variables()
    if options['model_name'] == 'quickdraw':
        actions = [action.set_p_selector(i, len(actions)) for i, action in enumerate(actions)]

    heuristics = load_heuristics(options['mode'], actions, model_wr, options['length'])
    search = SequenceSearch(model_wr, actions, heuristics, config=base_config)

    for idx, instance in enumerate(instances_oh.values):
        start = timeit.default_timer()
        if options['model_name'] == 'quickdraw':
            result = search.find_correction(instance.reshape((1, instance.shape[0], instance.shape[1])),
                                            np.array([target_prediction]), sess)
        else:
            result = search.find_correction(instance.reshape((1, instance.shape[0])),
                                            np.array([target_prediction]), sess)
        if result.best_result is not None:
            inst = processing.undummify(pd.DataFrame(instance.reshape((1, -1)), columns=data.get_feature_order()))
            inst[target_name] = np.argmax(model.model.predict(instance.reshape((1, -1))))
            # Get original Feature order for Measurements
            inst = inst[old_ordered_columns]
            cf = processing.undummify(pd.DataFrame(result.best_result.final_instance.reshape((1, -1)),
                                                    columns=data.get_feature_order()))
            cf[target_name] = np.argmax(model.model.predict(result.best_result.final_instance.reshape((1, -1))))
            # Get original Feature order for Measurements
            cf = cf[old_ordered_columns]

            if cf[target_name].values[0] != inst[target_name].values[0]:
                counterfactuals.append(cf)
            else:
                cf[:] = np.nan
                counterfactuals.append(cf)

            test_instances.append(inst)
        stop = timeit.default_timer()
        time_taken = stop - start
        times_list.append(time_taken)

    counterfactuals_df = pd.DataFrame(np.array(counterfactuals).squeeze(), columns=instances.columns)
    instances_df = pd.DataFrame(np.array(test_instances).squeeze(), columns=instances.columns)

    # Success rate & drop not successful counterfactuals & process remainder
    success_rate, counterfactuals_indeces = measure.success_rate_and_indices(
        counterfactuals_df[cont_features].astype('float64'))
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

    return instances_list, counterfactuals_list, times_list, success_rate
