import pandas as pd
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
    :return:
    """
    if label is not None:
        lbl = df.pop(label)

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    result = pd.DataFrame(x_scaled, columns=df.columns)

    if label is not None:
        result[label] = lbl

    return result


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


def one_hot_encode_instance(data, instance, categorical_features, target_name):
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
    indexed_data = indexed_data.drop(columns=target_name)
    indexed_data = pd.get_dummies(indexed_data, columns=categorical_features)

    encoded_instance = indexed_data.loc[indexed_data['idx'] == n_inst]

    return encoded_instance.drop(columns='idx')
