import pickle
import pandas as pd
import library.data_processing as process

filePath = '../Datasets/Adult/'
testFile = 'adult_test.csv'
trainFile = 'adult.csv'

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']
categorical_features = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                        'native-country']

# Read Data from csv
test_df = pd.read_csv(filePath + testFile, skipinitialspace=True, header=None)
train_df = pd.read_csv(filePath + trainFile, skipinitialspace=True, header=None)
# if data is already cleansed, don't do anything
if test_df.shape[1] <= len(column_names):
    test_df.columns = column_names
    train_df.columns = column_names

    # Concatenate test and train set to one dataframe to remain consistency
    df = pd.concat([test_df, train_df])
    # convert all '?' to NaN and drop those rows
    df = df[df != '?']
    df.dropna(axis=0, inplace=True)

    # convert binary features in 0 and 1
    # <=50K: 0; >50K: 1
    df['income'] = df['income'].str.rstrip('.')  # Original test data contains '.' as last element in column income
    process.categorize_binary(df, 'income', '>50K')

    # One hot encode all categorical features
    # df = pd.get_dummies(df, columns=categorical_features)

    # save cleansed data in edited folder
    df.to_csv(filePath + 'adult_full.csv', index=False)

    # save column names of categorical data
    with open(filePath + 'cat_names.txt', 'wb') as f:
        pickle.dump(categorical_features, f)
