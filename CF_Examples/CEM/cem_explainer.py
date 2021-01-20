import tensorflow as tf
import numpy as np
import pandas as pd

import ML_Model.ANN_TF.model_ann as model_tf
import CF_Models.cem_ml.utils as util
from CF_Models.cem_ml.aen_cem import AEADEN

import library.data_processing as preprocessing
import library.measure as measure
import timeit


def counterfactual_search(dataset_filename, data_name, instance, session, model, max_iter=100,
                       binary_steps=9, init_const=10, mode='PN', kappa=0, beta=1e-1, gamma=0.5):

        """
                Compute counterfactual for CEM
                :param dataset_filename: String; Filename (e.g. 'adult_full')
                :param instances: Dataframe; Instances to generate counterfactuals
                :param model: model; pretrained classifier (could in principle be anything)
                :param mode: String; one of {PN, PP}; PN corresponds to counterfactual explanation
                :param max_iter: int > 0;
                :param init_const: int > 0;
                :param kappa: float (?) > 0; defines the margin between the two classses; the higher kappa, the more certain we will be
                :param beta: float > 0; importance given to ell_1 term
                :param gamma: float > 0; importance given to AutoEncoder term
                """#
        # load the generation model: AE
        if data_name == 'adult':
            dataset_filename = dataset_filename.split('.')[0]
            AE_model = util.load_AE(dataset_filename)
            
        elif data_name == 'compas':
            dataset_filename = dataset_filename.split('.')[0]
            AE_model = util.load_AE(dataset_filename)

        orig_prob, orig_class, orig_prob_str = util.model_prediction(model, np.expand_dims(instance, axis=0))

        target_label = orig_class
        orig_sample, target = util.generate_data(instance, target_label)

        # start the search
        counterfactual_search = AEADEN(session, model, mode=mode, AE=AE_model, batch_size=1, kappa=kappa, init_learning_rate=1e-2,
                                        binary_search_steps=binary_steps, max_iterations=max_iter, initial_const=init_const, beta=beta,
                                        gamma=gamma)

        counterfactual = counterfactual_search.attack(orig_sample, target)

        adv_prob, adv_class, adv_prob_str = util.model_prediction(model, counterfactual)
        delta_prob, delta_class, delta_prob_str = util.model_prediction(model, orig_sample - counterfactual)

        INFO = "[kappa:{}, Orig class:{}, Adv class:{}, Delta class: {}, Orig prob:{}, Adv prob:{}, Delta prob:{}".format(
                kappa, orig_class, adv_class, delta_class, orig_prob_str, adv_prob_str, delta_prob_str)
        print(INFO)

        if np.argmax(model.model.predict(instance.reshape(1,-1))) != np.argmax(model.model.predict(counterfactual.reshape(1,-1))):
                counterfactual = counterfactual
        else:
                counterfactual = counterfactual
                counterfactual[:] = np.nan

        return instance, counterfactual.reshape(-1)


def get_counterfactual(dataset_path, dataset_filename, data_name, instances, binary_cols, continuous_cols, target_name, model, sess):

        """
        :param dataset_filename: str; Filename (e.g. 'adult_full')
        :param data_name: str; Data name (e.g. 'adult')
        :param instances: Dataframe; Instances to generate counterfactuals for
        :param binary_cols: list; list of features which need to be one_hot_encoded
        :param continuous_cols: list; list of numeric features
        :param target_name: str; target corresponding to data set
        :return: input instances & counterfactual explanations
        """  #

        # drop targets
        data = pd.read_csv(dataset_path + dataset_filename)
        data = data.drop(columns=[target_name])
        instances = instances.drop(columns=[target_name])

        # normalize instances
        instances = preprocessing.normalize_instance(data, instances, continuous_cols)
        # binary instances in robust way
        instances = preprocessing.robust_binarization(instances, binary_cols, continuous_cols)

        counterfactuals = []
        times_list = []

        for i in range(instances.values.shape[0]):
                start = timeit.default_timer()
                _, counterfactual = counterfactual_search(dataset_filename, data_name, instances.values[i, :], sess, model)
                stop = timeit.default_timer()
                time_taken = stop - start

                counterfactuals.append(counterfactual)
                times_list.append(time_taken)

        counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
        counterfactuals_df.columns = instances.columns

        # Success rate & drop not successful counterfactuals & process remainder
        success_rate, counterfactuals_indeces = measure.success_rate_and_indices(counterfactuals_df)
        counterfactuals_df = counterfactuals_df.iloc[counterfactuals_indeces]
        instances = instances.iloc[counterfactuals_indeces]

        # Obtain labels
        instance_label = np.argmax(model.model.predict(instances.values), axis=1)
        counterfactual_label = np.argmax(model.model.predict(counterfactuals_df.values), axis=1)

        # Round binary columns to integer
        counterfactuals_df[binary_cols] = counterfactuals_df[binary_cols].round(0).astype(int)

        # Order counterfactuals and instances in original data order
        counterfactuals_df = counterfactuals_df[data.columns]
        instances = instances[data.columns]

        # Convert binary cols of counterfactuals and instances into strings: Required for >>Measurement<< in script
        counterfactuals_df[binary_cols] = counterfactuals_df[binary_cols].astype("string")
        instances[binary_cols] = instances[binary_cols].astype("string")

        # Convert binary cols back to original string encoding
        counterfactuals_df = preprocessing.map_binary_backto_string(data, counterfactuals_df, binary_cols)
        instances = preprocessing.map_binary_backto_string(data, instances, binary_cols)

        # Add labels
        counterfactuals_df[target_name] = counterfactual_label
        instances[target_name] = instance_label

        # Collect in list making use of pandas
        instances_list = []
        counterfactuals_list = []

        for i in range(counterfactuals_df.shape[0]):
                counterfactuals_list.append(
                        pd.DataFrame(counterfactuals_df.iloc[i].values.reshape((1, -1)), columns=counterfactuals_df.columns))
                instances_list.append(pd.DataFrame(instances.iloc[i].values.reshape((1, -1)), columns=instances.columns))

        return instances_list, counterfactuals_list, times_list, success_rate
