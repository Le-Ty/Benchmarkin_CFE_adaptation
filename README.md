## Requirements
- Python 3.6 or higher  

## Installation
This project is intended to run on Windows or Linux.

### Install required Packages
All requirements can also be found in the *requirements.txt*.
````
make requirements
````

## Run Experiment
To reproduce one of our experiments, we recommend the following steps.
- Choose one of our benchmarking script
  - ``benchmarking.py`` 
  
- Comment all counterfactual methods that should not be evaluated.
- Results are saved to ``csv`` with call of ``df_results.to_csv(Path/to/save/result)``
- Run script

## Benchmark arbitrary counterfactual method
- Implement new model such that it outputs the factuals, counterfactuals, computation times and success rate
  - Factuals and counterfactuals are lists of ``Dataframes``
  - Succes rate is the fraction of how many counterfactual examples really flip the prediction of a black box model
  
- Call the ``compute_measurements(...)`` method from ``benchmarking.py`` with the following inputs
  - Dataframe of the whole data
  - List of Dataframes of original instances
  - List of Dataframes with counterfactual examples
  - List with continuous features
  - String with column name of the prediction label
  - ML model we want to analyse
  - List with column names of immutable features
  - List of computational times
  - Succes rate
  - Optional:
    - Boolean if factual and counterfactual are already normalized.
    - Boolean if factual and counterfactual should be one-hot-encoded (*true*) or binarized (*false*)
    - Boolean if factual and counterfactual are already encoded
    - String with a separator that should be used for one-hot-encoding

## Datasets
All datasets are already preprocessed and can be found in ``Datasets``.  
In the following we describe preprocessing steps for our datasets. All steps are implemented in ``Data_Cleansing``.

### Give Me Credit
Test dataset is missing information for column 'SeriousDlquin2yrs' which is used in the training set as label.  
All NaN-values are dropped. 

### Adult
- Missing information is encoded as '?'. Rows with this value are dropped
- Binary features 'income' are encoded with:
  - <=50K: 0, >50K: 1

Dropped the Education column as it seems to be redudant
- Most info should alread by captured by 'Education-num'

The categorical variables are kept as strings and encoded as follows:
- Nationality: US, Non US
- Maritial status: Husband, Non-Husband
- Occupation status: Specialist or Managerial, Non- (Specialist or Managerial)
- Race: White, Non-White
- Sex: Femal, Male

### COMPAS
- Drop columns id, name, first, last, c_case_number, r_case_number, vr_case_number, 
v_type_of_assessment, type_of_assessment, v_score_text, score_text, age_cat, dob, 
compas_screening_date, c_offense_date, c_arrest_date, num_r_cases, r_days_from_arrest,
r_offense_date, num_vr_cases, vr_charge_degree, vr_offense_date, vr_charge_desc, v_screening_date,
screening_date
- The categorical variables are kept as strings and encoded as follows:
  - race: White, Non-White
  - r_charge_degree and c_charge_degree: F=3, M=2, O=1
- jail_in and jail_out columns are compressed to one jail time column which results after enddate - startdate
  - jail time of only a few hours is converted to 1 day
  - NaN values (no jail time) are converted to 0


## Black Box Models
### Artificial Neural Network
- Tensorflow and Pytorch version of a Multi-Layer Perceptron, with 3 hidden layer of size 18, 9 and 3.
- ReLu activation function
- RMSProb optimization and Binary-Cross-Entropy-Loss
### Logistic Regression
- Tensorflow and Pytorch version of a Linear Model with no hidden layer and no activation function.
- RMSProb optimization and Binary-Cross-Entropy-Loss

## Counterfactual Methods
- [Counterfactual Latent Uncertainty Explanations(CLUE)](https://arxiv.org/pdf/2006.06848.pdf)
- [EB-CF](https://arxiv.org/pdf/1912.03277.pdf)
- [Feasible and Actionable Counterfactual Explanations (FACE)](https://arxiv.org/pdf/1909.09369.pdf)
