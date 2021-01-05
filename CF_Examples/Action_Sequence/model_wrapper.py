import tensorflow as tf

"""
This wrapper takes an arbitrary black box model and makes it for Action Sequence accessible
"""


# TODO: Works yet only with Tensorflow ann_tf

class Model_wrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        """
        Predict output based on input
        :param inputs: Input tensor
        :return: output tensor
        """
        reshaped_inputs = tf.reshape(inputs, (1, inputs.shape[1]))
        # return self.model.model.predict(reshaped_inputs, steps=1)
        return self.model.model(inputs)
