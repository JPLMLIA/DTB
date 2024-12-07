# MLUQ

MLUQ is a novel way to evaluate the accuracy of machine learning models and get better predictions that minimize maximum loss. Traditional methods use cross validation or a test set to evaluate the model giving a single score. In contrast MLUQ provides score with a confidence bound. Further, traditional ensemble methods use an average of all predictions (like random forest) as a prediction. In contrast, MLUQ uses a more complex algorithm using game theory that provides lower maximum loss.

## Loading the Conda environment
MLUQ comes with a prepared conda environment. The conda enviornment is stored in MLUQ.yml. A GPU is helpful for running this tensorflow version of MLUQ but is not required. The algorithm will be slower though.

1. Open a terminal and go to the directory containing the MLUQ.yml. 
2. Run "conda env create -f MLUQ.yml"

## get_estimator_probabilities
MLUQ for tensorflow is composed of 3 high level API functions. The first is get_estimator_probabilities.

This function takes in a dataset and performs a Min-Max decision theoretic game . It returns a set of trained estimators and their associated probabilities. The probabilities are essentially the weight the prediction should give to the estimator.

#### INPUTS
**CORES (integer >= 1):** The number of cores to use for parallelization. The defult is 1.\
**ray_start_function (function):** This is a function that contains the configuration for the ray.init function. See the ray project documentation for how to create a ray init function. An example is below:\
def ray_start_function():
    ray.init(num_cpus = 2, memory = 2000000000, object_store_memory = 2000000000)\
**data_input (pandas dataframe):** This is the data that will be used in comparisons (training, predicting, etc.). the data should be a pandas dataframe.\
**number_of_estimators (integer >= 1):** The number of estimators to use for prediction and probability assignment.\
**number_of_diracs (integer >= 1):** The data is split into diracs or sometimes called folds in this code. The number_of_diracs is the number of splits.\
**bootstrap_fraction (float <= 1.0):** The data is sampled in the comparisons. The bootstrap_fraction is the fraction to sample from the data\
**purification_fraction (float <= 1.0):** When finding the probabilities there is a feature that compares the game data to a set of test comparisons in the data. This fraction is what fraction to use.\
**purification_repetitions (integer >= 1):** The purification has to be run through repetitions. More repetitions creates more robust comparisons. More repetitions takes longer computation time.\
**create_compile_and_fit_function (function):** This is a function that creates compiles and fits the tensorflow model. It should take 2 parameters called X and y. It should return the fitted model. See the code below for an example.\
**test_fraction (float <= 1.0):**  The comparisons in each trial are done on a test set. This is the fraction from the total data to use as a test set.\
**target_column (string):** The name of the column that holds the target variable in the data_input.\

#### OUTPUTS
**decision_theoretic_models (dictionary):** A dictionary of trained models of type estimators_model. The keys are sequential integers.\
**PII_probabilities (list):** The probability weights for each estimator in decision_theoretic_models.\

```python
from MLUQRegressionTF import MLUQRegressionHighLevelTF

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import pandas as pd
import tensorflow as tf
import ray

#fetch the data
housing_data = fetch_california_housing()
#convert the data to a pandas dataframe. convert only features for now
data = pd.DataFrame(data = housing_data.data, columns = housing_data.feature_names)
#set the target for the dataframe.
data['__Target__'] = housing_data.target 
target_column = '__Target__'
#split the data into a train and test set
data_train, data_test = train_test_split(data, test_size = 0.2)

def create_compile_and_fit(X, y):
    #define the model
    model = tf.keras.Sequential([
		tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001) ),
		tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dense(1)
        ])

    #compile model
    model.compile(
	loss = 'mse',
	optimizer = tf.keras.optimizers.RMSprop(2.5e-4),
	metrics = ['mse'])

    #fit the model
    model.fit(
	x = X, 
	y = y, 
	epochs = 16, 
	verbose = 1,
	callbacks = None,
	batch_size = 64)

    return model


CORES = 10
def ray_start():
	ray.init(num_cpus = CORES, memory = 20000000000, object_store_memory = 20000000000)

decision_theoretic_models, PII_probabilities = MLUQRegressionHighLevelTF.get_estimator_probabilities(		
							CORES = CORES,
							ray_start_function = ray_start,
							data_input = data_train,
							number_of_estimators = 10,
							number_of_diracs = 100,
							bootstrap_fraction = float(0.005),
							purification_fraction = 0.2,
							purification_repetitions = 10,
							create_compile_and_fit_function = create_compile_and_fit,
							test_fraction = 0.2,
							loss = mean_squared_error,
							target_column = '__Target__')
```

```
Training 0 Out Of 10

Epoch 1/16
1/1 [==============================] - 0s 316us/step - loss: 66.5581 - mse: 66.5154
Epoch 2/16
1/1 [==============================] - 0s 297us/step - loss: 14.2509 - mse: 14.2084
Epoch 3/16
1/1 [==============================] - 0s 288us/step - loss: 6.2522 - mse: 6.2098
Epoch 4/16
1/1 [==============================] - 0s 293us/step - loss: 4.6502 - mse: 4.6079
Epoch 5/16
1/1 [==============================] - 0s 321us/step - loss: 4.2606 - mse: 4.2184
Epoch 6/16
1/1 [==============================] - 0s 328us/step - loss: 4.1138 - mse: 4.0717
...
...
...
(pid=18347) Running UQ 5 Out of 10
(pid=18344)
(pid=18344) Running UQ 3 Out of 10
(pid=18345)
(pid=18345) Running UQ 8 Out of 10
(pid=18349)
(pid=18349) Running UQ 2 Out of 10
(pid=18348)
(pid=18348) Running UQ 6 Out of 10
...
...
...
Training 1 Out Of 10
...
...
...
Training 9 Out Of 10
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
**loss (scikit learn loss function):** The loss function to evaluate the estimator predictions. Can be a scikit-learn function such as Mean Squared Error\
**target_column (string):** The name of the column that holds the target variable. The default is __Target__ but this will break if the target column is not __Target__. __Target__ will work if the default dataset is used in get_estimator_probabilities.\

#### OUTPUTS
**predictions (array like):** The predictions for the target on the data input.\
**loss (float):** The error between the predictions and the target using the loss function.\

```python
from MLUQRegressionTF import MLUQRegressionHighLevelTF

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import pandas as pd
import tensorflow as tf
import ray

#fetch the data
housing_data = fetch_california_housing()
#convert the data to a pandas dataframe. convert only features for now
data = pd.DataFrame(data = housing_data.data, columns = housing_data.feature_names)
#set the target for the dataframe.
data['__Target__'] = housing_data.target 
target_column = '__Target__'
#split the data into a train and test set
data_train, data_test = train_test_split(data, test_size = 0.2)

def create_compile_and_fit(X, y):
    #define the model
    model = tf.keras.Sequential([
		tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001) ),
		tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dense(1)
        ])

    #compile model
    model.compile(
	loss = 'mse',
	optimizer = tf.keras.optimizers.RMSprop(2.5e-4),
	metrics = ['mse'])

    #fit the model
    model.fit(
	x = X, 
	y = y, 
	epochs = 16, 
	verbose = 1,
	callbacks = None,
	batch_size = 64)

    return model


CORES = 10
def ray_start():
	ray.init(num_cpus = CORES, memory = 20000000000, object_store_memory = 20000000000)

decision_theoretic_models, PII_probabilities = MLUQRegressionHighLevelTF.get_estimator_probabilities(		
							CORES = CORES,
							ray_start_function = ray_start,
							data_input = data_train,
							number_of_estimators = 10,
							number_of_diracs = 100,
							bootstrap_fraction = float(0.005),
							purification_fraction = 0.2,
							purification_repetitions = 10,
							create_compile_and_fit_function = create_compile_and_fit,
							test_fraction = 0.2,
							loss = mean_squared_error,
							target_column = '__Target__')


predictions, loss = MLUQRegressionHighLevelTF.predict_using_decision_theoretic_probabilities(	
			CORES = CORES,
			ray_start_function = ray_start,
			data = data_test,
			estimators = decision_theoretic_models,
			PII_probabilities = PII_probabilities,
			loss = mean_squared_error,
			target_column = '__Target__')

print('predictions:', predictions)
print('loss:', loss)
```

```
Training 0 Out Of 10

1/1 [==============================] - 0s 332us/step - loss: 8333.8555 - mse: 8333.8135
Epoch 2/16
1/1 [==============================] - 0s 308us/step - loss: 7341.6978 - mse: 7341.6562
Epoch 3/16
1/1 [==============================] - 0s 304us/step - loss: 6678.2266 - mse: 6678.1851
Epoch 4/16
1/1 [==============================] - 0s 328us/step - loss: 6150.0425 - mse: 6150.0010
Epoch 5/16
1/1 [==============================] - 0s 342us/step - loss: 5701.9639 - mse: 5701.9224
Epoch 6/16
1/1 [==============================] - 0s 436us/step - loss: 5305.3599 - mse: 5305.3184
Epoch 7/16
1/1 [==============================] - 0s 365us/step - loss: 4949.7915 - mse: 4949.7500
Epoch 8/16
1/1 [==============================] - 0s 322us/step - loss: 4628.6841 - mse: 4628.6426
...
...
...
(pid=18347) Running UQ 5 Out of 10
(pid=18344)
(pid=18344) Running UQ 3 Out of 10
(pid=18345)
(pid=18345) Running UQ 8 Out of 10
(pid=18349)
(pid=18349) Running UQ 2 Out of 10
(pid=18348)
(pid=18348) Running UQ 6 Out of 10
...
...
...
Training 1 Out Of 10
...
...
...
Training 9 Out Of 10
...
...
...

```

```
predictions: [[2.488429 ]
 [3.2984755]
 [1.0069947]
 ...
 [2.2675898]
 [3.5560443]
 [2.293901 ]]
loss: 3.3326124688048475

```
