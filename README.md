# <center> NaiveBayesianClassifier </center>

Implementation of a Naive Bayesian Classifier for the Census Income or Adult dataset (http://archive.ics.uci.edu/ml/datasets/Adult)

Prediction task is to __determine whether a person makes over 50K a year__.

## Data visualization

TODO

## Data preprocessing

All the preprocessing steps are made by the `Preprocessor` class in `preprocessor.py`.

### Missing values  
This dataset contain missing values so I implemented 2 diffrent methods to deal with them:
- __Remove__ : if their is a missing value, we just remove the entire row
- __Average/most frequent__ : if their is a missing value, we replace it with the mean value of the column if it's a continous value or with the most frequent value in the column if it's a categorical data.  

You can choose between this 2 options when you create the preprocessor in `test.py`:
```Python
preprocessor = Precessor(missing_value_method="remove")
or
preprocessor = Precessor(missing_value_method="average")
```

### Continuous data
A other problem with the adult dataset is that it contains continous values whitch is a problem to use the Naive Bayes algorithm.  
To deal with thoses continuous values I used the equal-width binning with the most relevant bin I found during my data visualization step using Tableau.  
The bins I used :

| Continuous attribute | Size of the bin |
|----------------------|:---------------:|
| age                  | 5               |
| fnlwgt               | 22500           |
| hours-per-week       | 5               |

You can easly change those value by changing the `column2Descrtize` dictionary in `Preprocessor.process()`

## Naive Bayes

The Naive Bayes algorithm in implemented in the `NaiveBayes` class in `naiveBayes.py`.  
The algorithm use the conditional probability of each attribute knowing the label. All the probabilities `P(attribute=value | income='>50K')` and `P(attribute=value | income='<=50K')` are stored in the dictionary attribute `NaiveBayes.probabilities`.

## K-fold cross-Validation

TODO

## Results and Analysis

### Relevant attribute for classification

First we want to know if all the attribute we have in the dataset are relevant to predict the income. So for each attribute we tried to predict the income using only this attribute and we look at the accuracy. The results are listed in the following table:

| Attribute used for prediction | Accuracy |
|-------------------------------|:--------:|
| age                           | 59.67%   |
| workclass                     | 66.56%   |
| fnlwgt                        | 66.68%   |
| education                     | 71.95%   |
| education_num                 | 71.48%   |
| marital_status                | 71.08%   |
| occupation                    | 65.79%   |
| relationship                  | 71.33%   |
| race                          | 33.19%   |
| sex                           | 50.08%   |
| capital_gain                  | 79.37%   |
| capital_loss                  | 77.16%   |
| hours_per_week                | 70.09%   |
| native_country                | 29.18%   |

TODO : correlation with visualization  
TODO : most relevant and less relevant

### Classification results
