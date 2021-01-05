import json
import tensorflow as tf
import numpy as np
import pandas as pd

from CF_Models.act_seq.actions.feature import Feature, StrokeFeature, CategoricFeature, BaseFeature


def create_feature_mapping(df, target):
    """
    creates dictionary with feature meta info and its mapping according to Action Sequence
    :param df: dataframe with original data (not one-hot-encoded)
    :param target: String, with column name of target feature
    :return: tupel of feature and mapping dictionaries
    """
    raw_features = []
    mapping = dict()
    columns = df.columns

    i = 0  # Counter for each feature
    idx = 0  # Counter incl. feature values

    for col in columns:
        num_values = 1
        feature_col = df[col]
        feature = dict()

        feature['name'] = col
        feature['i'] = i
        feature['idx'] = idx

        if feature_col.dtypes == 'object':
            # Categorical data (without target feature) gets type nominal
            if col == target:
                feature['type'] = 'class'
            else:
                feature['type'] = 'nominal'

            # Encode String values with prefix i and counting suffix according to Action Sequence
            # every encoded value is saved in list values.
            # Mapping for encoding is saved in dictionary mapping
            values_without_duplicates = feature_col.values.tolist()
            values_without_duplicates = list(dict.fromkeys(values_without_duplicates))
            values = []
            suffix = 0
            mapping[col] = dict()
            num_values = 0  # new counting for each value
            for s in values_without_duplicates:
                encoded = str(i) + str(suffix)
                mapping[col][s] = int(encoded)
                values.append(int(encoded))

                suffix += 1
                num_values += 1
            feature['values'] = values

        elif col != target:
            mean = feature_col.mean()
            std = feature_col.std()

            feature['mean'] = mean
            feature['std'] = std
            feature['type'] = 'numeric'

        feature['num_values'] = num_values
        idx += num_values
        i += 1

        raw_features.append(feature)

    return raw_features, mapping


def loader(raw_features):
    """
    loader for numeric features similar to feature.py from Action Sequence to
    load features from list of dictionaries.
    :param raw_features: list of dict containing feature meta info
    :return: list of Feature objects
    """
    features = dict()
    input_dim = sum(
        [feature['num_values'] for feature in raw_features if feature['type'] != 'class'])
    for feature in raw_features:
        if feature['type'] == 'numeric':
            features[feature['name']] = Feature(feature['name'], feature['idx'], feature['mean'],
                                                feature['std'], input_dim)
        elif feature['type'] == 'stroke':
            features[feature['name']] = StrokeFeature(feature['name'], feature['idx'], input_dim)
        elif feature['type'] != 'class':
            features[feature['name']] = CategoricFeature(feature['name'], feature['idx'],
                                                         feature['values'], input_dim)
    return features
