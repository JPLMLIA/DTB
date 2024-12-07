
from .Utils import *

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os.path
from os import path
import sys

import tensorflow as tf

#
def get_estimator_probabilities(		
	CORES = 1,
	ray_start_function = None,
	data_input = None,
	number_of_estimators = 10,
	number_of_diracs = 100,
	bootstrap_fraction = float(0.005),
	purification_fraction = 0.2,
	purification_repetitions = 100,
	create_compile_and_fit_function = None,
	test_fraction = 0.2,
	loss = mean_squared_error,
	target_column = '__Target__'):
	
	'''

	This function builds a set of estimators and performs the MLUQ
	Min Max game to get the probabilities for each estimator. The estimators
	and respective probabilities are returned. 

	Inputs
		CORES (integer >= 1): The number of cores to use for parallelization. The defult is 1.

		ray_start_function (function): This is a function that contains the configuration
		for the ray.init function. See the ray project documentation for how to 
		create a ray function. An example is below:
		def ray_start_function():
			ray.init(num_cpus = 2, memory = 2000000000, object_store_memory = 2000000000)

		data_input (pandas dataframe): This is the data that will be used in comparisons
		(training, predicting, etc.). the data should be a pandas dataframe.

		number_of_estimators (integer >= 1): The number of estimators to use for prediction 
		and probability assignment.
		
		number_of_diracs (integer >= 1): The data is split into diracs or sometimes called
		folds in this code. The number_of_diracs is the number of splits.
		
		bootstrap_fraction (float <= 1.0): The data is sampled in the comparisons.
		The bootstrap_fraction is the fraction to sample from the data
		
		purification_fraction (float <= 1.0): When finding the probabilities there is a 
		a feature that compares the game data to a set of test comparisons
		in the data. This fraction is what fraction to use.  

		purification_repetitions (integer >= 1): The purification has to be run through 
		repetitions. More repetitions creates more robust comparisons. More
		repetitions takes longer computation time.

		create_compile_and_fit_function (function): This is a function that creates
		compiles and fits the tensorflow model. It should take 2 parameters called
		X and y. It should return the fitted model. An example is below:
		def create_compile_and_fit_function(X, y):
			#define the model
			model = tf.keras.Sequential([
					tf.keras.layers.Dense(128, activation = 'relu',
									kernel_regularizer = tf.keras.regularizers.l2(0.001) ),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
					tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
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

		test_fraction (float <= 1.0):  The comparisons in each trial are done on a test set.
		This is the fraction from the total data to use as a test set.

		target_column (string): The name of the column that holds the target variable
		in the data_input.


	Outputs
		decision_theoretic_models (dictionary): A dictionary of trained models of type
		estimators_model. The keys are sequential integers.
		
		PII_probabilities (list): The probability weights for each estimator
		in decision_theoretic_models. This is a list.
	'''	

	#start ray
	ray_start_function()

	#assign the data variable to the data_input input	
	data = data_input

	#split the data into train_UQ and test sets
	data = shuffle(data)
	data_train_UQ = data.iloc[ :int(len(data.index) * 0.8) ]
	data_test = data.iloc[ int(len(data.index) * 0.8): ]

	#delete the data for memory purposes
	del data

	#get a copy for the data_train_UQ
	data_train_UQ_copy = data_train_UQ.copy()

	#split the data into train and UQ sets
	data_train_UQ_copy = shuffle(data_train_UQ_copy)
	data_train = data_train_UQ_copy.iloc[ :int(len(data_train_UQ_copy.index) * 0.8) ]
	data_UQ = data_train_UQ_copy.iloc[ int(len(data_train_UQ_copy.index) * 0.8): ]

	#delete the datasets that are no longer needed
	del data_train_UQ_copy
	del data_train_UQ

	#sort the data train and UQ sets by the target_column.
	data_train.sort_values(target_column, inplace = True)
	data_train.reset_index(inplace = True, drop = True)

	data_UQ.sort_values(target_column, inplace = True)
	data_UQ.reset_index(inplace = True, drop = True)

	#train the estimators for the decision theoretic comparison
	decision_theoretic_models = estimator_training(
					data_train = data_train,
					number_of_estimators = number_of_estimators,
					bootstrap_fraction = bootstrap_fraction,
					loss = loss,
					create_compile_and_fit_function = create_compile_and_fit_function,
					target_column = target_column)

	#run the decision theoretic algorithm and get the probabilities for each
	#estimator. the PII probabilities are the estimator probabilities
	game_matrix, game_value, PII_probabilities = seperate_sort_dicracs(
							data_UQ = data_UQ,
							number_of_diracs = number_of_diracs,
							estimators = decision_theoretic_models,
							loss = loss,
							purification = [purification_fraction,
											purification_repetitions],
							target_column = target_column,
							detailed_game = False,
							CORES = CORES,
							ray_start_function = ray_start_function)		

	
	ray.shutdown()

	return decision_theoretic_models, PII_probabilities


def predict_using_decision_theoretic_probabilities(	
	CORES = 1,
	ray_start_function = None,
	data = None,
	estimators = None,
	PII_probabilities = None,
	loss = mean_squared_error,
	target_column = None,
	visualize_confidence_intervals = False,
	visualize_fraction = 0.02):

	'''
	This function uses the outputs and probabilities from 
	get_estimator_probabilities to predict on a dataset. The prediction
	and the error relative to the true values are returned.

	Inputs
		CORES (integer >= 1): The number of cores to use for parallelization. The defualt is 1.
		
		ray_start_function (function): This is a function that contains the configuration
		for the ray.init function. See the ray project documentation for how to 
		create a ray function. An example is below:
		def ray_start_function():
			ray.init(num_cpus = 2, memory = 2000000000, object_store_memory = 2000000000)
		
		data (pandas dataframe): The dataset to predict on.

		estimators (dictionary): A dictionary of estimators created using 
		get_estimator_probabilities.

		PII_probabilities (list): The relative probabilities for each estimator
		in estimators. This is an output from get_estimator_probabilities.

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as Mean Squared Error
	
	 	target_column (string): The name of the column that holds the target variable.
		 
		visualize_confidence_intervals (True or False): Whether to visualize the confidence
		intervals from the test set predictions

		visualize fraction (float between 0.0 and 1.0 inclusive): The fraction of the 
		test data to visualize using confidence intervals. Only relevant if 
		visualize_confidence_intervals is True
	Outputs:
		predictions (array like): The predictions for the target on the data input.

		error (float): The error between th predictions and the target using
		the loss error function.
	'''

	#start ray
	ray_start_function()

	predictions, error = decision_predict_parallel(
				data_test = data,
				estimators = estimators,
				PII_probabilities = PII_probabilities,
				loss = loss,
				target_column = target_column,
				CORES = 1,
				visualize_confidence_intervals = visualize_confidence_intervals,
				visualize_fraction = visualize_fraction)

	ray.shutdown()

	return predictions, error