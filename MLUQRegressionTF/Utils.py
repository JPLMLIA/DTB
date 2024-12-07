
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from multiprocessing import Pool
from sklearn.utils import shuffle

import tensorflow as tf
import random
import ray
import copy 
import os
from os import path
import matplotlib.pyplot as plt

"""
@author: peyman
@author: danny

Backend for machine learning uncertainty quantification using a game 
theoretic approach.
"""


def PII_view_solver(
	A, 
	METHOD = 'interior-point', 
	VERBOSE = False):

	'''
	#FIXME

	The function is part of the min-max game that is used to optimize the
	probability distributions of each estimator. The game is structured
	as a game between 2 players. This is the PII solver. 

	Inputs
		A:
		METHOD:
		VERBOSE: True or False. True outputs more information.

	Outputs
		W:
		q:
	
	'''

	m, n = A.shape
	c = -1.0 * np.ones(n)
	A_ub = A
	b_ub = np.ones(m)
	bounds = [(0,None)]*n
	LP = linprog(
		c,
		A_ub = A_ub,
		b_ub = b_ub,
		bounds = bounds,
		method = METHOD)

	print('Success: ', LP.success)
	if VERBOSE: print(LP)
	W = -1.0/LP.fun
	q = W*LP.x
	return W, q


def PI_view_solver(
	A, 
	METHOD = 'interior-point', 
	VERBOSE = False):

	'''
	#FIXME

	The function is part of the min-max game that is used to optimize the
	probability distributions of each estimator. The game is structured
	as a game between 2 players. This is the PI solver. 
	#
	Inputs
		A:
		METHOD:
		VERBOSE: True or False. True outputs more information.

	Outputs
		W:
		q:
	
	'''

	m, n  = A.shape
	c = np.ones(m)
	A_ub = -1.0*A.T
	b_ub = -1.0*np.ones(n)
	bounds = [(0,None)]*m
	LP = linprog(
		c,
		A_ub=A_ub,
		b_ub=b_ub,
		bounds=bounds,
		method=METHOD)
	print('Success: ', LP.success)
	if VERBOSE: print(LP)
	V = 1.0/LP.fun
	p = V*LP.x

	return V, p

def estimator_training(
	data_train,
	number_of_estimators,
	bootstrap_fraction,
	loss,
	create_compile_and_fit_function,
	target_column = None):

	'''
	trains number_of_estimators of type model.

	Inputs:
		data_train (pandas dataframe): The data to train the model on

		number_of_estimators (integer >= 1): The number of models to train.

		bootstrap_fraction (float <= 1.0): Each estimator is trained using a bootstrap sample
		from the data_train. Bootstrap fraction is the fraction to sample from
		data_train.

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

		target_column (string): the name of the target column in data_train


	Outputs:
		estimators (dictionary): a dictionary mapping the estimator number to the 
		scikit-learn model.
	'''

	estimators = {}

	for i in range(number_of_estimators):
		#sample from the data_train set
		data_sample = data_train.sample(frac = bootstrap_fraction)
		#get the features
		X = data_sample.drop(columns = [target_column])
		#get the targets
		y = data_sample[target_column]

		#print progress
		print("Training", i + 1 , "Out Of", number_of_estimators)

		model = create_compile_and_fit_function(X, y)

		#save model to path
		list_of_files = os.listdir('./KModels')
		if len(list_of_files) == 0:
			model_path = 'KModels/' + str(0)
		else:
			list_of_files = [int(filename) for filename in list_of_files]
			next_filename = max(list_of_files) + 1 
			model_path = 'KModels/' +  str(next_filename)
		model.save(model_path)

		#save path in dictionary
		estimators[i] = model_path

		#clear tensorflow global session
		tf.keras.backend.clear_session()

	return estimators


def game_matrix_build(
	game_data,
	number_of_diracs,
	estimators,
	loss,
	target_column):

	'''

	This function builds a game matrix where each estimator is evaluated
	against a subset of the game_data. 

	Inputs
		game_data (pandas dataframe): The data that is used in playing the game between P I and 
		P II. Contains a column called __Fold__ that holds a number 
		indicating the what fold each row belongs to.

		number_of_diracs (integer >= 1): The number of folds in the data
		
		estimators (dictionary): A dictionary containing the estimators that were trained
		using estimator_training 

		loss (scikit learn loss function): The loss function to use when playing the game. The loss function
		can be a scikit-learn function like mean_squared_error

		target_column (string): The name of the column that holds the targets.

	Outputs:
		game_matrix (numpy array): The completed game_matrix
	'''

	#create a zero matrix to start the game
	game_matrix = np.zeros((number_of_diracs, len(estimators)))

	list_of_models = {}
	for i in estimators:
		list_of_models[i] = tf.keras.models.load_model(estimators[i])


	#for every fold in the game data
	for fold in game_data['__Fold__'].unique():
		#get the fold data from the game_data
		fold_data = game_data.loc[ game_data['__Fold__'] == fold ]
		#get the features of the fold data
		X = fold_data.drop(columns = [target_column, '__Fold__'])
		#get the targets of the targets of the fold data
		y = fold_data[target_column]
		
		for estimator_index in estimators:
			model = list_of_models[estimator_index]


			#get the loss from predicting on the fold data using the estimator 
			#at index i in the estimators. this is the score for that estimator  
			game_matrix[fold, estimator_index] = (
				loss(y, model.predict(X)) )

	#return the game matrix
	return game_matrix

@ray.remote
def seperate_sort_dicracs_parallel_helper(
	data_UQ, 
	number_of_diracs, 
	estimators, 
	loss, 
	target_column, 
	repetitions, 
	counter):

	'''
	This function is a helper function to make the seperate_sort_dicracs
	function run in parallel. The function calls the game_matrix_build
	function on a single core.

	Inputs:
		data_UQ (pandas dataframe): The data to build the game matrix with.
		
		number_of_diracs (integer >= 1): The number of folds in the data
		
		estimators (dictionary): The dictionary of estimators that was obtained from 
		estimator_training
		
		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as Mean Squared Error
		
		target_column (string): The name of the column that contains the targets.
		
		repitition (integer >= 1): The number of times to perform purification. In this
		function it is used for a progress bar. 
		
		counter (integer): A number that is passed in to keep track of the progress
		of the parallel computation.

	Outputs:
		game_matrix (numpy array): The completed game_matrix

		game_value (float): The value of the game from the PII view

		PII_probabilities (list): The probabilities assigned to each estimator
		in the estimators.

	'''


	#output a counter to keep track like a progress bar
	print()
	print("Running UQ", counter + 1, "Out of", repetitions)

	#build the game matrix
	game_matrix = game_matrix_build(
			data_UQ,
			number_of_diracs,
			estimators,
			loss,
			target_column)

	#solve the game from the PII view
	game_value, PII_probabilities = PII_view_solver(game_matrix)

	return [game_matrix, game_value, PII_probabilities]


def seperate_sort_dicracs(
	data_UQ,
	number_of_diracs,
	estimators,
	loss = mean_squared_error,
	purification = False,
	target_column = None,
	detailed_game = True,
	CORES = 1,
	ray_start_function = None):

	'''
	This function solves the game using the decision theoretic approach. 
	Purification is used if specified.
	
	
	Inputs:
		data_UQ (pandas dataframe): This is the data that was partitioned using train_UQ_split

		number_of_diracs (integer >= 1): The number of folds in the data

		estimators (dictionary): The dictionary of estimators that was obtained from 
		estimator_training

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as Mean Squared Error

		purification (list): A list containing a fraction (float <= 1.0) and the number of 
		repetitions (integer >= 1). If specified, the function run repetitions of building
		the game matrix. The repetitions are used to create a more robust 
		result

		target_column (string): The name of the column that contains the targets.

		detailed_game (True or False): Whether to output the game values and 
		PII probabilities as lists. 

	Outputs:
		There are 2 potential outputs in this function:

		(final_game_matrix, 
		final_game_value, 
		final_PII_probability) 
		
		and 
		
		(final_game_matrix, 
		final_game_value, 
		final_PII_probability, 
		historical_game_values, 
		historical_PII_probabilities)


		final_game_matrix (numpy array): The game matrix for the game. If purification is 
		specified it is the average of the game matrices in the purification reps
		
		final_game_value (float): The value of the game. If purification is specified
		it is is the average of the game values in the purification reps

		final_PII_probabilities (list): The probabilities assigned to each estimator based
		on the performance in the game matrix. If purification is specified
		it is the average of the PII probabilities in the purification reps

		historical_game_values (list): The game_values for the game at each purification
		rep. This is a list that will have the same size as the number of 
		purification reps. 

		historical_PII_probabilities (list): The game_values for the game at each 
		purification rep. This is a list that will have the same size as the 
		number of purification reps.  

	'''

	#syntax sugar for constants
	FRACTION = 0 
	REPETITIONS = 1

	#create a column in data_UQ that partitions the data based on the number of 
	#diracs.
	data_UQ['__Fold__'] = ( 
		(data_UQ.index.values * number_of_diracs) // len(data_UQ) )


	#if purification is not None
	if purification != None:
		#get the sampling fraction and the repititions
		fraction = purification[FRACTION]
		repetitions = purification[REPETITIONS]

		#initialize a game matrix
		final_game_matrix = np.zeros( (number_of_diracs, len(estimators)) )
		#initialize a numpy array to hold the PII probabilities 
		final_PII_probabilities = np.zeros( len(estimators) )
		#initialize the game value
		final_game_value = 0.0

		#initialize the parallel arguments
		parallel_arguments = []

		
		#for every repitition
		for counter in range(repetitions):

			#when sampling, there is a chance that the sample will be empty for
			#some folds. the function checks if any folds are empty and
			#resamples if any are

			#a boolean that holds if any of the data folds are empty
			cleared_data = False

			#initialize the data to use in the game matrix build
			data_UQ_sample = None

			#while the data is not cleared
			while not cleared_data:
				#sample from the data_UQ input
				data_UQ_sample = data_UQ.sample(frac = fraction)

				#check each fold in data_UQ_sample to see if there is any fold
				#that is empty
				data_folds_empty = []
				for fold in data_UQ_sample['__Fold__'].unique():
					data_folds_empty.append( 
						not any(data_UQ_sample['__Fold__'] == fold) )

				#if any data fold is empty
				if any(data_folds_empty):
					cleared_data = False
				else:
					cleared_data = True

			#append the parallel arguments that will be used to call the 
			#parallel helper function.  it is unpredictable in what order
			#the results arguments will be processed and in what order
			#the results will be.
			parallel_arguments.append([
				data_UQ_sample, 
				number_of_diracs, 
				estimators, 
				loss, 
				target_column, 
				repetitions, 
				counter])

		#initialize lists for the results from calling a parallel helper function
		parallel_results = []
		parallel_results_phase = []

		#for every arguments in parallel_arguments
		for i in range( len(parallel_arguments) ):
			#call the helper function as a future (see ray documentation)
			parallel_results_phase.append( seperate_sort_dicracs_parallel_helper.remote(*parallel_arguments[i]) )

			#if the number of futures is equal to the number of cores or
			#the last list of arguments has been called. This is done to 
			#limit the memory consumption of the program. The memory is cleared
			#after every phase. A phase is defined as running one argument on 
			#each core available. If the number number of cores is 10 then 
			#10 arguments would be run in parallel then the memory would be cleared
			#and a new phase would start with the next set of 10
			if i % CORES == 0 or i == ( len(parallel_arguments) - 1 ):
				#run the futures in parallel
				parallel_results.extend( copy.deepcopy( ray.get(parallel_results_phase) ) )
				#print a counter
				print('list of results iteration', i, ':', len(parallel_results))
				#clear the list of futures for the phase
				parallel_results_phase.clear()
				#reset ray to clear memory
				ray.shutdown()
				ray_start_function()

		#syntax sugar for constants
		GAME_MATRIX = 0
		GAME_VALUE = 1
		PII_PROBABILITIES = 2 

		parallel_game_matrices = []
		parallel_game_values = []
		parallel_PII_probabilities = []

		#for every result in parallel result
		for i in range( len(parallel_results) ):
			#parse the game matrix, game value and PII probabilities
			parallel_game_matrices.append( parallel_results[i][GAME_MATRIX] )
			parallel_game_values.append( parallel_results[i][GAME_VALUE] )
			parallel_PII_probabilities.append( parallel_results[i][PII_PROBABILITIES] )

		#the total game matrix can be constructed by summing each game matrix
		#that was returned from the parallel helper. the same is true for the 
		#game values and PII probabilities. the game value, PII probability and 
		# the game matrix for the total run are the averages of the runs from 
		# the purification reps
		final_game_value = sum(parallel_game_values) / len(parallel_game_values)
		final_PII_probabilities = sum(parallel_PII_probabilities) / len(parallel_PII_probabilities)
		
		final_game_matrix = parallel_game_matrices[0]
		for i in range( 1, len(parallel_game_matrices) ):
			final_game_matrix = final_game_matrix + parallel_game_matrices[i]
		
		final_game_matrix = final_game_matrix / len(parallel_game_matrices)


		#if the inputs request detailed return values
		if detailed_game:
			#construct a data frame to hold all the game values from each
			#purification run
			historical_game_values = pd.DataFrame()
			historical_game_values['Game Values'] = np.array(parallel_game_values)

			historical_PII_probabilities = parallel_PII_probabilities

			return  (final_game_matrix, 
				final_game_value, 
				final_PII_probabilities, 
				historical_game_values, 
				historical_PII_probabilities)
		else:
			return  (final_game_matrix, 
				final_game_value, 
				final_PII_probabilities)

		
	#if there is no purification then a single game matrix 
	#can be built in parallel
	else:
		final_game_matrix = Game_Matrix_Build(
					data_UQ,
					number_of_diracs,
					estimators,
					loss,
					target_column)

		final_game_value, final_PII_probabilities = PII_view_solver(final_game_matrix)

	return final_game_matrix, final_game_value, final_PII_probabilities

@ray.remote
def decision_predict_parallel_helper(
	estimator_index, 
	estimators, 
	X, 
	y,
	PII_probabilities, 
	counter, 
	total):
	'''
	This function is used to parallelize decision_predict_parallel. 

	Inputs:
		estimator_index (integer): The index of the estimator in the estimators obtained
		from estimator_training
		
		estimators (dictionary): The dictionary of estimators that was obtained from 
		estimator_training

		X (pandas dataframe): The features of the data to predict on

		y (pandas dataframe): The targets of the data that will be used in prediction

		PII_probabilities (list): The probabilities assigned to each model in 
		estimators. The probabilities are found from seperate_sort_dicracs.

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as Mean Squared Error

		counter (integer): An integer passed in to keep track of the progress in the 
		parallel computation.

		total (integer): The total number of times that decision_predict_parallel_helper
		will be called.

	Outputs:
		prediction (array like): The prediction of the estimator on the features multiplied
		by its probability in PII_probabilities
	'''

	print("Predicting", counter + 1, "Out of", total)

	model = tf.keras.models.load_model(estimators[estimator_index])

	prediction = model.predict(X) * PII_probabilities[estimator_index]	
	
	return prediction 


def decision_predict_parallel(
	data_test = None,
	estimators = None,
	PII_probabilities = None,
	loss = mean_squared_error,
	target_column = None,
	CORES = 1,
	visualize_confidence_intervals = False,
	visualize_fraction = 0.02):

	'''

	This function predicts using the estimators and probabilities obtained
	using seperate_sort_dicracs.

	Inputs:
		data_test (pandas dataframe): The test data to evaluate the PII_probabilities

		estimators (dictionary): The dictionary of estimators that was obtained from 
		estimator_training

		PII_probabilities (list): The probabilities assigned to each model in 
		estimators. The probabilities are found from seperate_sort_dicracs.

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as Mean Squared Error

		target_column (string): The name of the column that holds the target variable. 
		The default is __Target__

		CORES (integer >= 1): The number of cores to run in parallelization
		
		visualize_confidence_intervals (True or False): Whether to visualize the confidence
		intervals from the test set predictions

		visualize fraction (float between 0.0 and 1.0 inclusive): The fraction of the 
		test data to visualize using confidence intervals. Only relevant if 
		visualize_confidence_intervals is True
	Outputs:
		final_prediction (array like): The predictions on the test data using the 
		PII_probabilities

		test_loss (scikit learn loss function): The less function applied to the final_predictions and
		the targets. It is the error for the predictions using the 
		PII_probabilities
	'''


	#get the features from the test data
	X = data_test.drop(columns = [target_column])
	#get the targets from the test data
	y = data_test[target_column]

	#initialize arguments to call multiprocessing
	parallel_arguments = []
	#initialize counter for progress bar
	i = 0
	#get total number of estimators
	total = len(estimators)

	for estimator_index in estimators:		
		#append the arguments that will be passed to the helper function
		#and processed in parallel. it is unpredictable in what order
		#the results arguments will be processed and in what order
		#the results will be.
		parallel_arguments.append([ 
			estimator_index, 
			estimators, 
			X, 
			y, 
			PII_probabilities,  
			i, 
			total])

		#update the counter
		i = i + 1

	#initialize lists for the results from calling a parallel helper function
	parallel_results = []
	parallel_results_phase = []

	#call the helper function as a future (see ray documentation)
	for i in range( len(parallel_arguments) ):
		parallel_results_phase.append( decision_predict_parallel_helper.remote(*parallel_arguments[i]) )
	#run all the parallel arguments in parallel using available cores 
	parallel_results.extend( copy.deepcopy( ray.get(parallel_results_phase) ) )

	#The final prediction is the sum of the results of the parallel helper
	#the loss is the loss function applied to the test targets and the 
	#final predictions
	final_prediction = parallel_results[0]
	for i in range( 1, len(parallel_results) ):
		final_prediction = final_prediction + parallel_results[i]
	test_loss = loss(y, final_prediction)

	if visualize_confidence_intervals:
		#syntax sugar
		ROW = 0
		COLUMN = 1

		#get the confidence intervals as the standard deviation using 
		#the prediction mean
		square_difference = pd.DataFrame(columns = [ str(decision) for decision in range( len(estimators.keys()) )])

		for estimator_index in estimators:
			model = tf.keras.models.load_model(estimators[estimator_index])
			prediction = model.predict(X) * PII_probabilities[estimator_index]
			prediction = prediction[0]
			square_difference[str(estimator_index)] = ((prediction - y)**2) * PII_probabilities[estimator_index]
		confidence_intervals = np.sqrt(square_difference.sum(axis = COLUMN))

		#make a dataframe that contains the truth, mean and confidence interval
		confidence_intervals_data = pd.DataFrame(columns = ['truth','mean','CI'])
		confidence_intervals_data['truth'] = y
		confidence_intervals_data['mean'] = final_prediction
		confidence_intervals_data['CI'] = confidence_intervals

		#sample from the data a fraction for visualization
		confidence_intervals_data = confidence_intervals_data.sample(frac = visualize_fraction)

		#get the x axis as a range of numbers
		x_axis = range( len(confidence_intervals_data['truth']) ) 

		#plot 2 plots. Each shows the same information in different ways
		plt.figure()
		plt.scatter(
			x_axis, 
			confidence_intervals_data['truth'], 
			marker = '.',
			color = 'k', 
			label = "Ground Truth")

		plt.errorbar(
			x_axis,
			confidence_intervals_data['mean'],
			yerr = confidence_intervals_data['CI'],
			fmt = '.',
			label = 'Prediction Mean')

		plt.fill_between(
			x_axis,
			(confidence_intervals_data['mean'] - confidence_intervals_data['CI']),
			(confidence_intervals_data['mean'] + confidence_intervals_data['CI']),
			color = 'b',
			alpha = 0.1,
			linewidth = 2,
			label = 'Confidence Intervals')

		plt.legend()
		plt.xlabel('Datapoint number')
		plt.ylabel('Target')
		plt.title('Prediction vs Truth with Confidence Intervals')


		plt.figure()
		plt.plot(
			x_axis, 
			confidence_intervals_data['truth'], 
			marker = '',
			color = 'k', 
			linewidth = 1,
			label = "Ground Truth")

		plt.plot(
			x_axis, 
			confidence_intervals_data['mean'], 
			marker = '',
			color = 'r', 
			linewidth = 1,
			label = "Prediction Mean")

		plt.scatter(
			x_axis, 
			confidence_intervals_data['truth'], 
			marker = '.',
			color = 'k', 
			label = "Training Points")

		plt.fill_between(
			x_axis,
			(confidence_intervals_data['mean'] - confidence_intervals_data['CI']),
			(confidence_intervals_data['mean'] + confidence_intervals_data['CI']),
			color = 'b',
			alpha = 0.1,
			linewidth = 2,
			label = 'Confidence Intervals')

		plt.legend()
		plt.xlabel('Datapoint number')
		plt.ylabel('Target')
		plt.title('Prediction vs Truth with Confidence Intervals')

		plt.show()

	return final_prediction, test_loss
