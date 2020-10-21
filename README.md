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