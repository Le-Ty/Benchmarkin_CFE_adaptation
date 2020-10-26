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
