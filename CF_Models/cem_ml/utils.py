## utils.py -- Some utility functions
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from keras.models import Model, model_from_json, Sequential

import tensorflow as tf
import os
import numpy as np


def load_AE(data_name, print_summary=True):

    saveFilePrefix = "CF_models/cem_ml/AE_models/Saved_models/"

    # model is saved as a .json & weights are saved as h5
    decoder_model_filename = saveFilePrefix + "ae_tf_" + data_name + ".json"
    decoder_weight_filename = saveFilePrefix + "ae_tf_" + data_name + ".h5"

    if not os.path.isfile(decoder_model_filename):
        raise Exception("The file for decoder model does not exist:{}".format(decoder_model_filename))
    json_file = open(decoder_model_filename, 'r')
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    if not os.path.isfile(decoder_weight_filename):
        raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
    decoder.load_weights(decoder_weight_filename)

    if print_summary:
        print("Decoder summaries")
        decoder.summary()

    return decoder


def generate_data(instance, target_label):
    inputs = []
    target_vec = []

    inputs.append(instance)
    target_vec.append(np.eye(2)[target_label])  # 2: since we only look at binary classification

    inputs = np.array(inputs)
    target_vec = np.array(target_vec)

    return inputs, target_vec

def model_prediction(model, inputs):
    prob = model.model.predict(inputs)
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace('\n','')
    return prob, predicted_class, prob_str
