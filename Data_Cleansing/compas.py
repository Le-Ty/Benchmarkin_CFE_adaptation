import pickle
import pandas as pd
import library.data_processing as process

filePath = '../Datasets/COMPAS/'
fileName = 'compas-scores.csv'

# Read Data from csv
compas_df = pd.read_csv(filePath + fileName, skipinitialspace=True)

# if data is already cleansed, don't do anything
if compas_df.shape[1] == 47:
    # drop columns which are not necessary for learning
    columns_to_drop = ['id', 'name', 'first', 'last', 'c_case_number', 'r_case_number', 'vr_case_number',
                       'v_type_of_assessment', 'type_of_assessment', 'v_score_text', 'score_text', 'age_cat', 'dob',
                       'compas_screening_date', 'c_offense_date', 'c_arrest_date', 'num_r_cases', 'r_days_from_arrest',
                       'r_offense_date', 'num_vr_cases', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
                       'v_screening_date', 'screening_date', 'c_charge_desc', 'r_charge_desc']
    if compas_df.columns.isin(columns_to_drop).any():
        compas_df = compas_df.drop(columns_to_drop, axis=1)

    # # convert binary features in 0 and 1
    # # male: 0; female: 1
    # process.categorize_binary(compas_df, 'sex', 'Female')

    # convert jail_in and jail_out columns to jail time columns in days
    compas_df['c_jail_time'] = (
            pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    compas_df['r_jail_time'] = (
            pd.to_datetime(compas_df['r_jail_out']) - pd.to_datetime(compas_df['r_jail_in'])).dt.days
    compas_df = compas_df.drop(['c_jail_in', 'c_jail_out'], axis=1)
    compas_df = compas_df.drop(['r_jail_in', 'r_jail_out'], axis=1)
    # jail time lower than 1 day but greater than 0 are converted to 1 day
    compas_df.loc[compas_df['c_jail_time'] == 0, 'c_jail_time'] += 1
    compas_df.loc[compas_df['r_jail_time'] == 0, 'r_jail_time'] += 1
    # no jail time (NaN) is converted to 0
    compas_df['c_jail_time'] = compas_df['c_jail_time'].fillna(0)
    compas_df['r_jail_time'] = compas_df['r_jail_time'].fillna(0)

    # One hot encode all categorical features
    categorical_features = ['race', 'sex', 'is_violent_recid']
    # compas_df = pd.get_dummies(compas_df, columns=categorical_features)

    # simplify categorical features
    race_map = {
        'Caucasian': 'White',
        'African-American': 'Non-White',
        'Asian': 'Non-White',
        'Hispanic': 'Non-White',
        'Native American': 'Non-White',
        'Other': 'Non-White'
    }
    compas_df['race'] = compas_df['race'].map(race_map)

    charge_degree_mape = {
        'O': 1,
        'M': 2,
        'F': 3
    }
    compas_df['r_charge_degree'] = compas_df['r_charge_degree'].map(charge_degree_mape)
    compas_df['c_charge_degree'] = compas_df['c_charge_degree'].map(charge_degree_mape)

    compas_df = compas_df.rename(columns={'decile_score.1': 'decile_score_2'})
    compas_df['is_violent_recid'].loc[compas_df['is_violent_recid'] == 0] = 'Low'
    compas_df['is_violent_recid'].loc[compas_df['is_violent_recid'] == 1] = 'High'

    # Drop rows without information
    compas_df.dropna(axis=0, inplace=True)

    # save column names of categorical data
    with open(filePath + 'cat_names.txt', 'wb') as f:
        pickle.dump(categorical_features, f)

# save cleansed data in edited folder
compas_df.to_csv(filePath + fileName, index=False)
