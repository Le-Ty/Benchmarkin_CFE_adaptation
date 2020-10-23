import pandas as pd
import library.data_processing as process

filePath = '../Datasets/Home_Credit_Default_Risk/'
testFile = 'application_test.csv'
trainFile = 'application_train.csv'

# Read Data from csv
test_df = pd.read_csv(filePath + testFile, skipinitialspace=True)
train_df = pd.read_csv(filePath + trainFile, skipinitialspace=True)

# if data is already cleansed, don't do anything
if test_df.shape[1] == 121:
    # Concatenate test and train set to one dataframe to remain consistency
    train_df['set'] = 'train'
    test_df['set'] = 'test'
    test_df['TARGET'] = -1  # No Label data for test set => Dummy Value -1
    df = pd.concat([test_df, train_df])

    # drop columns which are not necessary or have to less information
    columns_to_drop = ['SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_3',
                       'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                       'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
                       'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
                       'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
                       'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
                       'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
                       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
                       'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
                       'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE',
                       'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
    if df.columns.isin(columns_to_drop).any():
        df = df.drop(columns_to_drop, axis=1)

    # convert binary features in 0 and 1
    # Cash loans: 0; Revolving loans: 1
    process.categorize_binary(df, 'NAME_CONTRACT_TYPE', 'Revolving loans')
    # M: 0; F:1
    process.categorize_binary(df, 'CODE_GENDER', 'F')
    # N:0; Y:1
    process.categorize_binary(df, 'FLAG_OWN_CAR', 'Y')
    process.categorize_binary(df, 'FLAG_OWN_REALTY', 'Y')

    # One hot encode all categorical features
    categorical_features = ['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
    df = pd.get_dummies(df, columns=categorical_features)

    # Encode Weekdays with int
    encodings = {
        "WEEKDAY_APPR_PROCESS_START": {
            "MONDAY": 0,
            "TUESDAY": 1,
            "WEDNESDAY": 2,
            "THURSDAY": 3,
            "FRIDAY": 4,
            "SATURDAY": 5,
            "SUNDAY": 6
        }
    }
    df = df.replace(encodings)

    # drop rows without missing information ( only 1/3 of rows remain)
    df = df.dropna(axis=0)

    # Split df back into test and train set
    test_df = df.loc[df['set'] == 'test']
    test_df = test_df.drop('set', axis=1)
    test_df = test_df.drop('TARGET', axis=1)
    train_df = df.loc[df['set'] == 'train']
    train_df = train_df.drop('set', axis=1)

    # # save cleansed data in edited folder
    test_df.to_csv(filePath + testFile, index=False)
    train_df.to_csv(filePath + trainFile, index=False)
