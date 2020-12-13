# Benchmarkin_Counterfactual_Examples  

## Datasets
=> Due to problems with already one-hot-encoded data for Counterfactual-Models, one-hot-encoding should only be 
done shortly before training.
### German Credit Score  
There are two versions of this data. The file german_numeric consists of already transformed data.  

### Give Me Credit
Test dataset is missing information for column 'SeriousDlquin2yrs' which is used in the training set as label.  
All NaN-values are dropped. 

### HELOC  
This dataset consists of the following special values    

- -9: No bureau record or no investigation
- -8: No usable/valid trades or inquiries
- -7: Condition not met (e.g. no inquiries, no delinquancies)

The preprocessing from 'https://github.com/Trusted-AI/AIX360/blob/master/aix360/datasets/heloc_dataset.py'
treats them as follows:
- all rows with only -9 are dropped and all remaining special values are declared as NaN
- or all special values are set to default value 0

Since the default value of 0 is mostly irritating (zero values could mean for a column like 
'Month since most recent delinquency' that there is a recent delinquency, where there is no delinquency
for special value -7), we choosed the following default values:
- Rows which contains only -9 values are dropped
- All cells with -7 are receiving max values of its column, to avoid the problem mentioned above
- Cells with -8, -9 are set to average values of its class (good, bad) and column

### Adult
- Missing information is encoded as '?'. Rows with this value are dropped
- Binary features 'income' are encoded with:
  - <=50K: 0, >50K: 1

### Australian
- Small dataset, no cleansing necessary

### COMPAS
- Drop columns id, name, first, last, c_case_number, r_case_number, vr_case_number, 
v_type_of_assessment, type_of_assessment, v_score_text, score_text, age_cat, dob, 
compas_screening_date, c_offense_date, c_arrest_date, num_r_cases, r_days_from_arrest,
r_offense_date, num_vr_cases, vr_charge_degree, vr_offense_date, vr_charge_desc, v_screening_date,
screening_date
- Binary features 'sex' and 'income' are encoded with:
  - male: 0; female: 1
- jail_in and jail_out columns are compressed to one jail time column which results after enddate - startdate
  - jail time of only a few hours is converted to 1 day
  - NaN values (no jail time) are converted to 0
- one hot encoding for race, c_charge_desc, r_charge_degree, c_charge_degree, r_charge_desc

### Home Credit Default Risk
- Drop columns SK_ID_CURR
  - Drop additional columns which contain to less information: EXT_SOURCE_1, EXT_SOURCE_3,
APARTMENTS_AVG,	BASEMENTAREA_AVG, YEARS_BEGINEXPLUATATION_AVG, YEARS_BUILD_AVG, 
COMMONAREA_AVG,ELEVATORS_AVG, ENTRANCES_AVG, FLOORSMAX_AVG, FLOORSMIN_AVG, LANDAREA_AVG, 
LIVINGAPARTMENTS_AVG, LIVINGAREA_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG, APARTMENTS_MODE, 
BASEMENTAREA_MODE, YEARS_BEGINEXPLUATATION_MODE, YEARS_BUILD_MODE, COMMONAREA_MODE, ELEVATORS_MODE, 
ENTRANCES_MODE, FLOORSMAX_MODE, FLOORSMIN_MODE, LANDAREA_MODE, LIVINGAPARTMENTS_MODE, 
LIVINGAREA_MODE, NONLIVINGAPARTMENTS_MODE, NONLIVINGAREA_MODE, APARTMENTS_MEDI, BASEMENTAREA_MEDI,
YEARS_BEGINEXPLUATATION_MEDI, YEARS_BUILD_MEDI, COMMONAREA_MEDI, ELEVATORS_MEDI, ENTRANCES_MEDI, 
FLOORSMAX_MEDI, FLOORSMIN_MEDI, LANDAREA_MEDI, LIVINGAPARTMENTS_MEDI, LIVINGAREA_MEDI, 
NONLIVINGAPARTMENTS_MEDI, NONLIVINGAREA_MEDI, FONDKAPREMONT_MODE, HOUSETYPE_MODE, TOTALAREA_MODE,
WALLSMATERIAL_MODE, EMERGENCYSTATE_MODE
- Binary features encoded as follows:
  - NAME_CONTRACT_TYPE: Cash loans: 0; Revolving loans: 1
  - FLAG_OWN_CAR/ FLAG_OWN_REALTY: N: 0; Y: 1
- One hot encoding for NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS,
NAME_HOUSING_TYPE, OCCUPATION_TYPE, ORGANIZATION_TYPE
- Encoding WEEKDAY_APPR_PROCESS_START as follows:
  - Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3, Friday: 4, Saturday: 5, Sunday: 6

## ML-Models
### ANN
- Neural Network with 2 hidden layers and Relu-Activation-Function 

## Counterfactual Methods
### MACE
- Requirement: With 'pysmt-install --z3' we have to install the z3 solver.