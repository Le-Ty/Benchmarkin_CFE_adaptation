import pandas as pd
import numpy as np

from sklearn import preprocessing


def categorize_binary(df, col, true_val):
    """
    Categorizes one column with two values into 0, 1 values
    :param df: dataframe object
    :param col: String column with binary data
    :param true_val: String value which should be declared as 1
    :return: inplace change in dataframe
    """
    if (col in df) and (df[col].dtype == 'object'):
        df['temp'] = 0
        df.loc[df[col].isin([true_val]), 'temp'] = 1
        df[col] = df['temp']
        df[col].astype('int64')
        df.drop('temp', axis=1, inplace=True)


def normalize(df, label=None):
    """
    Normalize each column of data between 0 and 1 except the label column
    :param df: dataframe to normalize
    :param label: column name of Label
    :param cont_features: list of featueres to normalize
    :return:
    """
    data = df.copy()

    # Get rid of data which should not be normalized
    if label is not None:
        lbl = data.pop(label)
    num_cols = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]

    result = data.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    result[num_cols] = min_max_scaler.fit_transform(result[num_cols])

    if label is not None:
        result[label] = lbl

    return result


def normalize_instance(df, instance, features):
    """
    Normalize a single instance
    :param df: Dataframe to fit scaler
    :param instance: Dataframe to normalize
    :param features: Continuous features to normalize
    :return: Dataframe with normalized continuous features
    """
    normalized_instance = instance.copy()

    # Define and fit Min-Max-Scaler
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df[features])

    normalized_instance[features] = scaler.transform(normalized_instance[features])

    return normalized_instance


def get_categorical_features(columns, continuous_features, label):
    """
    Computes list of categorical features
    :param columns: np.array of all features
    :param continuous_features: list of continuous features
    :param label: String with label name
    :return: np.array of categorical features
    """
    cat_features = []
    for feature in columns:
        if (feature not in continuous_features) and feature != label:
            cat_features.append(feature)

    return cat_features


def one_hot_encode_instance(data, instance, categorical_features):
    """
    one-hot-encode instance with respect to data to maintain consistency
    :param data: dataframe with whole dataset
    :param instance: dataframe to encode in context of data
    :param: categorical_features: list
    :param: target_name: String with label name
    :return: dataframe with encoded instance
    """
    n = data.shape[0]
    n_inst = n + 1
    index = range(n)

    indexed_data = data.copy()
    indexed_data['idx'] = index
    encoded_instance = instance.copy()
    encoded_instance['idx'] = n_inst

    indexed_data = indexed_data.append(encoded_instance)
    indexed_data = pd.get_dummies(indexed_data, columns=categorical_features)

    encoded_instance = indexed_data.loc[indexed_data['idx'] == n_inst]

    return encoded_instance.drop(columns='idx')


def robust_binarization(instances, binary_cols, continuous_cols):
    """
    robust processing: when binary feature only contains 1s or 0s, pd.get_dummies does neither one-hot encode
    properly nor does it binarize properly; thus, we need to make sure that binarization is correct
    :param instances:
    :param binary_cols:
    :param continuous_cols:
    :return: df including numeric variables + numeric binary variables
    """

    robust_names = continuous_cols + binary_cols

    instances = pd.get_dummies(instances, prefix_sep="__", columns=binary_cols, drop_first=True)
    non_robust_n = list(instances.columns)
    non_robust_names = []
    for i in range(len(non_robust_n)):
        prefix = non_robust_n[i].split('__')[0]
        non_robust_names.append(prefix)
    instances.columns = non_robust_names

    # Add missing columns
    for col in binary_cols:
        if col not in non_robust_names:
            print("Adding missing feature {}".format(col))
            instances[col] = 1

    # Make sure cols are in right order (first numeric; then binary)
    instances = instances[robust_names]

    return instances


def undummify(df, prefix_sep="_"):
    """
    Reverses one-hot-encoded data made by get_dummies from pandas
    :param df: Dataframe
    :param prefix_sep: String with Separator
    :return: Dataframe
    """
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df
