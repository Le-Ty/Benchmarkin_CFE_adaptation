# Benchmarkin_Counterfactual_Examples  

## Datasets
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
- Binary features 'sex' and 'income' are encoded with:
  - male: 0; female: 1
  - <=50K: 0, >50K: 1
- Categorical features with more than two categories are one-hot-encoded

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
