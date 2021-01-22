import numpy as np
import library.data_processing as processing

"""
This wrapper allows to use Action Sequence data object without loading from disk
and to use arbitrary datasets
"""


class Data_wrapper(object):
    def __init__(self, df, label, cat_feat, cont_feat, separator='_'):
        data_enc = processing.one_hot_encode_instance(df, df, cat_feat, separator=separator)
        data_enc = processing.normalize_instance(data_enc, data_enc, cont_feat)
        data_enc = data_enc.drop(label, axis=1)

        self.data = data_enc.values
        self.feature = data_enc.columns.values.tolist()

        # Action Sequence uses prob labels
        labels_1d = df[label].values
        labels_2d = np.array([1 - labels_1d, labels_1d]).reshape((-1, 2))
        self.label = labels_2d.tolist()

    def get_feature_order(self):
        """
        Returns columns of dataframe object in correct order.
        :return: List of column names without target label
        """
        return self.feature
