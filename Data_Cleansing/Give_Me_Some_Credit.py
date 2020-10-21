import pandas as pd

filePath = '../Datasets/Give_Me_Some_Credit/'
testFile = 'cs-test.csv'
trainFile = 'cs-training.csv'

# Read Data from csv
test_df = pd.read_csv(filePath + testFile, index_col=0)
train_df = pd.read_csv(filePath + trainFile, index_col=0)

# Column 'SeriousDlqin2yrs' only used in training set as label
try:
    test_df.drop('SeriousDlqin2yrs', axis=1, inplace=True)
except KeyError:
    print('Column "SeriousDlqin2yrs" already droped')

# drop rows with missing values
test_df.dropna(axis=0, inplace=True)
train_df.dropna(axis=0, inplace=True)

# save cleansed data in edited folder
test_df.to_csv(filePath + testFile, index=False)
train_df.to_csv(filePath + trainFile, index=False)
