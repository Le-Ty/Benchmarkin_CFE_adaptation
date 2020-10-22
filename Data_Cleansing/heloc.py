import pandas as pd
import library.data_processing as process

filePath = '../Datasets/HELOC/'
fileName = 'heloc_dataset_v1.csv'

# Read Data from csv
heloc_df = pd.read_csv(filePath + fileName)

# change column 'RiskPerformance' in Bad = 0 and Good = 1
process.categorize_binary(heloc_df, 'RiskPerformance', 'Good')

# Drop rows with value -9 in every column
label_risk = heloc_df.pop('RiskPerformance')
label_risk = label_risk[(heloc_df > -9).any(axis=1)]
heloc_df = heloc_df.loc[(heloc_df > -9).any(axis=1)]
heloc_df['RiskPerformance'] = label_risk

# Drop all columns with special values. Warning: only 2,500 rows will remain
# heloc_df = heloc_df[heloc_df >= 0]
# heloc_df.dropna(axis=0, inplace=True)

# set remaining special values with default values
# Max Month for values with -7
# Average values of its specific class for remaining special values
x_cols = list(heloc_df.columns.values)

for col in x_cols:
    # compute max and average values of specific column
    max_value = heloc_df[col].max()
    avg_value_good = heloc_df.loc[(heloc_df['RiskPerformance'] == 1) & (heloc_df[col] >= 0), col].mean()
    avg_value_bad = heloc_df.loc[(heloc_df['RiskPerformance'] == 0) & (heloc_df[col] >= 0), col].mean()
    # set max values for all columns with value -7
    heloc_df.loc[heloc_df[col] == -7, col] = max_value
    heloc_df.loc[(heloc_df['RiskPerformance'] == 0) & (heloc_df[col].isin([-8, -9])), col] = avg_value_bad
    heloc_df.loc[(heloc_df['RiskPerformance'] == 1) & (heloc_df[col].isin([-8, -9])), col] = avg_value_good

# save cleansed data in edited folder
heloc_df.to_csv(filePath + fileName, index=False)


