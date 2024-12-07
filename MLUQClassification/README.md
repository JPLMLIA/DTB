# MLUQ

MLUQ is a novel way to evaluate the accuracy of machine learning models and get better predictions that minimize maximum loss. Traditional methods use cross validation or a test set to evaluate the model giving a single score. In contrast MLUQ provides score with a confidence bound. Further, traditional ensemble methods use an average of all predictions (like random forest) as a prediction. In contrast, MLUQ uses a more complex algorithm using game theory that provides lower maximum loss.

## Loading the Conda environment
MLUQ comes with a prepared conda environment. The conda enviornment is stored in MLUQ.yml

1. Open a terminal and go to the directory containing the MLUQ.yml. 
2. Run "conda env create -f MLUQ.yml"

## get_estimator_probabilities
MLUQ is composed of 4 high level API functions. The first is get_estimator_probabilities.

This function takes in a dataset and performs a Min-Max decision theoretic game . It returns a set of trained estimators and their associated probabilities. The probabilities are essentially the weight the prediction should give to the estimator.

#### INPUTS
**CORES (integer >=1):** The number of cores to use for parallelization. The defult is 1.\
**number_of_estimators (integer >=1):** The number of estimators to use for prediction and probability assignment.\
**number_of_diracs (integer >=1):** The data is split into diracs or sometimes called folds in this code. The number_of_diracs is the number of splits.\
**bootstrap_fraction (float <= 1.0):** The data is sampled in the comparisons. The bootstrap_fraction is the fraction to sample from the data\
**purification_fraction (float <= 1.0):** When finding the probabilities there is a feature that compares the game data to a set of test comparisons in the data. This fraction is what fraction to use.\
**purification_repetitions (integer >=1):** The purification has to be run through repetitions. More repetitions creates more robust comparisons. More
repetitions takes longer computation time.\
**estimators_model (scikit-learn model):** The type of model to train. This must be a scikit-learn model. For Tensorflow, see the TF folder.\
**test_fraction (float <= 1.0):**  The comparisons in each trial are done on a test set. This is the fraction from the total data to use as a test set.\
**data_input (pandas dataframe):** This is the data that will be used in comparisons (training, predicting, etc.). If left blank as the default then the calfornia housing dataset from scikit learn will be used. Otherwise the data should be a pandas dataframe.\
**target_column (string):** The name of the column that holds the target variable in the data_input.\


#### OUTPUTS
**decision_theoretic_models (dictionary):** A dictionary of trained models of type estimators_model. The keys are sequential integers.\
**PII_probabilities (list):** The probability weights for each estimator in decision_theoretic_models.\

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

from MLUQClassification import MLUQClassificationHighLevel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#syntax sugar
ROW = 0
COLUMN = 1

#fetch housing data
CA_housing = fetch_california_housing()
#get housing features dataset
data_features = pd.DataFrame(data = CA_housing.data, columns = CA_housing.feature_names)
#set the targets as the house age
data_targets = data_features['HouseAge']
#drop house age from features
data_features = data_features.drop(['HouseAge'], axis = COLUMN)
#converts floating point targets to integer classes
data_targets_as_classes = [int(x) for x in data_targets]
#the data becomes the data features
data = data_features.copy()
#set the target as the house age classes
data['__Target__'] = data_targets_as_classes

#split the data into train and test sets
data_train, data_test = train_test_split(data,
                                        stratify = data['__Target__'], 
                                        test_size = 0.2)

decision_theoretic_models, PII_probabilities = MLUQClassificationHighLevel.get_estimator_probabilities(			
                                                        CORES = 2,
                                                        number_of_estimators = 10,
                                                        number_of_diracs = 100,
                                                        bootstrap_fraction = float(0.5),
                                                        purification_fraction = 0.2,
                                                        purification_repetitions = 100,
                                                        estimators_model = DecisionTreeClassifier(max_depth = 10),
                                                        test_fraction = 0.2,
                                                        loss = accuracy_score,
                                                        data_input = data_train,
                                                        target_column = '__Target__')
```

```
Training 1 Out Of 10
Training 2 Out Of 10
Training 3 Out Of 10
Training 4 Out Of 10
Training 5 Out Of 10
Training 6 Out Of 10
Training 7 Out Of 10
Training 8 Out Of 10
Training 9 Out Of 10
Training 10 Out Of 10


Running UQ 14 Out of 100
Running UQ 1 Out of 100
Success:  True

Running UQ 2 Out of 100
Success:  True

Running UQ 15 Out of 100
Success:  True

Running UQ 3 Out of 100
Success:  True

Running UQ 16 Out of 100
Success:  True

Running UQ 4 Out of 100
Success:  True

Running UQ 17 Out of 100
Success:  True
...
...
...
```


## predict_using_decision_theoretic_probabilities 
This function uses the outputs and probabilities from get_estimator_probabilities to predict on a dataset. The prediction and the loss from the true values are returned.


#### INPUTS
**CORES (integer >= 1):** The number of cores to use for parallelization. The defult is 1.\
**data (pandas dataframe):** The dataset to predict on.\
**estimators (dictionary):** A dictionary of estimators created using get_estimator_probabilities.\
**PII_probabilities (list):** The relative probabilities for each estimator in estimators. This is an output from get_estimator_probabilities.\
**loss (scikit learn loss function):** The loss function to evaluate the estimator predictions. Can be a scikit-learn function such as accuracy_score\
**target_column (string):** The name of the column that holds the target variable. The default is __Target__ but this will break if the target column is not __Target__. __Target__ will work if the default dataset is used in get_estimator_probabilities.\

#### OUTPUTS
**predictions (list):** The predictions for the target on the data input.\
**loss (float):** The error between the predictions and the target using the loss function.\

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

from MLUQClassification import MLUQClassificationHighLevel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#syntax sugar
ROW = 0
COLUMN = 1

#fetch housing data
CA_housing = fetch_california_housing()
#get housing features dataset
data_features = pd.DataFrame(data = CA_housing.data, columns = CA_housing.feature_names)
#set the targets as the house age
data_targets = data_features['HouseAge']
#drop house age from features
data_features = data_features.drop(['HouseAge'], axis = COLUMN)
#converts floating point targets to integer classes
data_targets_as_classes = [int(x) for x in data_targets]
#the data becomes the data features
data = data_features.copy()
#set the target as the house age classes
data['__Target__'] = data_targets_as_classes

#split the data into train and test sets
data_train, data_test = train_test_split(data,
                                        stratify = data['__Target__'], 
                                        test_size = 0.2)

decision_theoretic_models, PII_probabilities = MLUQClassificationHighLevel.get_estimator_probabilities(			
                                                        CORES = 2,
                                                        number_of_estimators = 10,
                                                        number_of_diracs = 100,
                                                        bootstrap_fraction = float(0.5),
                                                        purification_fraction = 0.2,
                                                        purification_repetitions = 100,
                                                        estimators_model = DecisionTreeClassifier(max_depth = 10),
                                                        test_fraction = 0.2,
                                                        loss = accuracy_score,
                                                        data_input = data_train,
                                                        target_column = '__Target__')


predictions, loss = MLUQClassificationHighLevel.predict_using_decision_theoretic_probabilities(
                        CORES = 2,
			data = data_test,
			estimators = decision_theoretic_models,
			PII_probabilities = PII_probabilities,
			loss = accuracy_score,
			target_column = '__Target__')

print("predictions:", predictions)
print('loss:', loss)
```

```
Training 1 Out Of 10
Training 2 Out Of 10
Training 3 Out Of 10
Training 4 Out Of 10
Training 5 Out Of 10
Training 6 Out Of 10
Training 7 Out Of 10
Training 8 Out Of 10
Training 9 Out Of 10
Training 10 Out Of 10

Running UQ 1 Out of 100

Running UQ 14 Out of 100
Success:  True

Running UQ 2 Out of 100
Success:  True

Running UQ 15 Out of 100
Success:  True

Running UQ 16 Out of 100
Success:  True

Running UQ 3 Out of 100
Success:  True
Success:  True


Running UQ 17 Out of 100
Running UQ 4 Out of 100
Success:  True
...
...
...

Predicting 1 Out of 10
Predicting 2 Out of 10
Predicting 3 Out of 10
Predicting 4 Out of 10
Predicting 5 Out of 10
Predicting 6 Out of 10
Predicting 7 Out of 10
Predicting 8 Out of 10
Predicting 9 Out of 10
Predicting 10 Out of 10
predictions: [25, 43, 41, 23, 17,  ... 44, 29, 40, 38, 30, 29]
loss: 0.04893410852713178
```
