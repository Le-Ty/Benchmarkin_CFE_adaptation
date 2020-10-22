import pandas as pd

filePath = '../Datasets/Australian/'
fileName = 'australian.csv'

columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']

# Read Data from csv
australian_df = pd.read_csv(filePath + fileName, delimiter=' ', names=columns)

# save cleansed data in edited folder
australian_df.to_csv(filePath + fileName, index=False)
