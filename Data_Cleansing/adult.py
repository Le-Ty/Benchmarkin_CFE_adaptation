import pandas as pd
import library.data_processing as process

filePath = '../Datasets/Adult/'
testFile = 'adult_test.csv'
trainFile = 'adult.csv'

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                        'native-country']

# Read Data from csv
test_df = pd.read_csv(filePath + testFile, skipinitialspace=True, header=None)
train_df = pd.read_csv(filePath + trainFile, skipinitialspace=True, header=None)
# if data is already cleansed, don't do anything
if test_df.shape[1] <= len(column_names):
    test_df.columns = column_names
    train_df.columns = column_names

    # Concatenate test and train set to one dataframe to remain consistency
    train_df['set'] = 'train'
    test_df['set'] = 'test'
    df = pd.concat([test_df, train_df])
    # convert all '?' to NaN and drop those rows
    df = df[df != '?']
    df.dropna(axis=0, inplace=True)

    # convert binary features in 0 and 1
    # male: 0; female: 1
    process.categorize_binary(df, 'sex', 'Female')
    # <=50K: 0; >50K: 1
    df['income'] = df['income'].str.rstrip('.')  # Original test data contains '.' as last element in column income
    process.categorize_binary(df, 'income', '>50K')

    # One hot encode all categorical features
    df = pd.get_dummies(df, columns=categorical_features)

    # Split df back into test and train set
    test_df = df.loc[df['set'] == 'test']
    test_df = test_df.drop('set', axis=1)
    train_df = df.loc[df['set'] == 'train']
    train_df = train_df.drop('set', axis=1)

    # save cleansed data in edited folder
    test_df.to_csv(filePath + testFile, index=False)
    train_df.to_csv(filePath + trainFile, index=False)
