
from .Utils import *

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os.path
from os import path
import sys

def compare_to_naive_average(
	CORES = 1,
	number_of_estimators = 10,
	number_of_diracs = 100,
	bootstrap_fraction = float(0.005),
	purification_fraction = 0.2,
	purification_repetitions = 100,
	estimators_model = DecisionTreeRegressor(max_depth = 20),
	number_of_trials = 30,
	test_fraction = 0.2,
	name_to_save = None,
	loss = mean_squared_error,
	data_input = None,
	target_column = '__Target__' ):


	'''
	This function runs number_of_trials comparisons between MLUQ and
	naive average. Each trial builds an ensemble of estimators_model with number_of_estimators.
	The ensemble is then tested on a test set with the decision theoretic 
	probabilities for each estimator and the naive average probabilities for
	each model. 

	Inputs
		CORES (integer >= 1): The number of cores to use for parallelization. The defult is 1.
		
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

		estimators_model (scikit-learn model):  The type of model to train. This must be a 
		scikit-learn model. For Tensorflow, see the TF folder.

		number_of_trials (integer >= 1): The number of comparisons to make between naive 
		average and MLUQ.

		test_fraction (float <= 1.0):  The comparisons in each trial are done on a test set.
		This is the fraction from the total data to use as a test set.

		loss (scikit-learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as Mean Squared Error
		
		data_input (pandas dataframe): This is the data that will be used in comparisons
		(training, predicting, etc.).

		target_column (string): The name of the column that holds the target variable
		in the data_input.
	'''

	#lists for comparison metrics
	maximum_folds_loss_decision_theoretic = []
	maximum_folds_loss_naive_average = []
	overall_average_loss_naive_average = []
	overall_average_loss_decision_theoretic = []
	
	for trial in range(number_of_trials):
		print( 'Running trial:', str(trial), 'out of', str(number_of_trials) )

		#the game data
		data = None

		data = data_input

		#split the data into train_UQ and test sets
		data = shuffle(data)
		data_train_UQ = data.iloc[ :int(len(data.index) * 0.8) ]
		data_test = data.iloc[ int(len(data.index) * 0.8): ]

		#delete the data for memory purposes
		del data
		
		#train the single model. trains one estimator using the 
		#all the data train_UQ
		single_model = estimator_training(
				data_train = data_train_UQ,
				model = estimators_model,
				number_of_estimators = 1,
				bootstrap_fraction = 1.0,
				target_column = target_column)

		#get a copy fo the data_train_UQ
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
						model = estimators_model,
						number_of_estimators = number_of_estimators,
						bootstrap_fraction = bootstrap_fraction,
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
								CORES = CORES)

		
		#make a copy of the test data
		data_test_copy = data_test.copy()
		data_test_copy.sort_values(target_column, inplace=True)
		data_test_copy.reset_index(inplace=True, drop=True)
		
		#split the test data into folds
		data_test_copy['__Fold__'] = (data_test_copy.index.values * number_of_diracs) // len(data_test_copy)
		
		#construct the game matrix using the test data
		test_game_matrix = game_matrix_build(
					game_data = data_test_copy,
					number_of_diracs = number_of_diracs,
					estimators = decision_theoretic_models,
					loss = loss,
					target_column = target_column)

		#make probabilities that will average predictions when run through the
		#decision predict functions.
		naive_average_probabilities = np.ones(PII_probabilities.shape)
		naive_average_probabilities = naive_average_probabilities / len(naive_average_probabilities)

		#predict on the test data using the decision theoretc probabilties
		#for each estimator
		decision_theoretic_prediction, decision_theoretic_error = decision_predict_parallel(
										data_test = data_test,
										estimators = decision_theoretic_models,
										PII_probabilities = PII_probabilities,
										loss = loss,
										target_column = target_column,
										CORES = CORES)

		#predict on the test data using the single model
		single_model_prediction, single_model_error = decision_predict_parallel(
								data_test = data_test,
								estimators = single_model,
								PII_probabilities = np.array( [1.0] ),
								loss = loss,
								target_column = target_column,
								CORES = CORES)

		#predict on the test data using a naive average probabilties for each 
		#estimator. a naive average approach is simulated by using 
		#decision theoretic probabilities that have equal weight
		naive_average_prediction, naive_average_error = decision_predict_parallel(
									data_test = data_test,
									estimators = decision_theoretic_models,
									PII_probabilities = naive_average_probabilities,
									loss = loss,
									target_column = target_column,
									CORES = CORES)


		#the maximum folds loss can be found by multiplying each estimator in the 
		#the test game matrix by its probability and finding the maximum. this is
		#true because the test game matrix holds the error for each estimator
		maximum_folds_loss_decision_theoretic.append(
			np.max(np.matmul(test_game_matrix, PII_probabilities)) )

		maximum_folds_loss_naive_average.append(
			np.max(np.matmul(test_game_matrix, naive_average_probabilities)) )

		#the overall average loss is the average overall loss when predicting
		#on the entire test data 
		overall_average_loss_decision_theoretic.append(decision_theoretic_error)
		overall_average_loss_naive_average.append(naive_average_error)

def get_estimator_probabilities(		
	CORES = 1,
	number_of_estimators = 10,
	number_of_diracs = 100,
	bootstrap_fraction = float(0.005),
	purification_fraction = 0.2,
	purification_repetitions = 100,
	estimators_model = DecisionTreeRegressor(max_depth = 20),
	test_fraction = 0.2,
	loss = mean_squared_error,
	data_input = None,
	target_column = '__Target__'):
	
	'''

	This function builds a set of estimators and performs the MLUQ
	Min Max game to get the probabilities for each estimator. The estimators
	and respective probabilities are returned. 

	Inputs
		CORES (integer >=1): The number of cores to use for parallelization. The defult is 1.
		
		number_of_estimators (integer >=1): The number of estimators to use for prediction 
		and probability assignment.
		
		number_of_diracs (integer >=1): The data is split into diracs or sometimes called
		folds in this code. The number_of_diracs is the number of splits.
		
		bootstrap_fraction (float <= 1.0): The data is sampled in the comparisons.
		The bootstrap_fraction is the fraction to sample from the data
		
		purification_fraction (float <= 1.0): When finding the probabilities there is a 
		a feature that compares the game data to a set of test comparisons
		in the data. This fraction is what fraction to use.  

		purification_repetitions (integer >=1): The purification has to be run through 
		repetitions. More repetitions creates more robust comparisons. More
		repetitions takes longer computation time.

		estimators_model (scikit-learn model) = The type of model to train. This must be a 
		scikit-learn model. For Tensorflow, see the TF folder.

		test_fraction (float <= 1.0):  The comparisons in each trial are done on a test set.
		This is the fraction from the total data to use as a test set.

		data_input (pandas dataframe): This is the data that will be used in comparisons
		(training, predicting, etc.).

		target_column (string): The name of the column that holds the target variable
		in the data_input.

	Outputs
		decision_theoretic_models (dictionary): A dictionary of trained models of type
		estimators_model. The keys are sequential integers.
		
		PII_probabilities (list): The probability weights for each estimator
		in decision_theoretic_models.
	'''	

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
					model = estimators_model,
					number_of_estimators = number_of_estimators,
					bootstrap_fraction = bootstrap_fraction,
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
							CORES = CORES)		

	return decision_theoretic_models, PII_probabilities

def predict_using_decision_theoretic_probabilities(	
	CORES = 1,
	data = None,
	estimators = None,
	PII_probabilities = None,
	loss = mean_squared_error,
	target_column = '__Target__',
	visualize_confidence_intervals = False,
	visualize_fraction = 0.0001):

	'''
	This function uses the outputs and probabilities from 
	get_estimator_probabilities to predict on a dataset. The prediction
	and the error relative to the true values are returned.

	Inputs
		CORES (integer >= 1): The number of cores to use for parallelization. The defult is 1.

		data (pandas dataframe): The dataset to predict on.

		estimators (dictionary): A dictionary of estimators created using 
		get_estimator_probabilities.

		PII_probabilities (list): The relative probabilities for each estimator
		in estimators. This is an output from get_estimator_probabilities.

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as Mean Squared Error
	
	 	target_column (string): The name of the column that holds the target variable. 
		The default is __Target__.

		visualize_confidence_intervals (True or False): Whether to visualize the confidence
		intervals from the test set predictions

		visualize fraction (float between 0.0 and 1.0 inclusive): The fraction of the 
		test data to visualize using confidence intervals

	Outputs:
		predictions (list): The predictions for the target on the data input.

		error (float): The error between th predictions and the target using
		the loss error function.
	'''

	predictions, error = decision_predict_parallel(
				data_test = data,
				estimators = estimators,
				PII_probabilities = PII_probabilities,
				loss = loss,
				target_column = target_column,
				CORES = 1,
				visualize_confidence_intervals = visualize_confidence_intervals,
				visualize_fraction = visualize_fraction)

	return predictions, error