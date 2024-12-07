
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from multiprocessing import Pool
from sklearn.utils import shuffle
from sklearn.utils import resample	

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
		A: The game matrix.
		METHOD: The linear programming (LP) numerical method used to solve the minimax.
		VERBOSE: True or False. True outputs more information.

	Outputs
		W: The value of the game (Nash equilibrium Value)
		q: Nash equilibrium mixed strategy (probabilities)
	
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

def estimator_training(
	data_train,
	model,
	number_of_estimators,
	bootstrap_fraction,
	target_column = '__Target__'):

	'''
	rains number_of_estimators of type model.

	Inputs:
		data_train (pandas dataframe): The data to train the model on

		model (scikit learn model): The type of model to train. This must be a scikit-learn model.
		For Tensorflow, see the TF folder.

		number_of_estimators (integer >= 1): The number of models to train.

		bootstrap_fraction (float <= 1.0): Each estimator is trained using a bootstrap sample
		from the data_train. Bootstrap fraction is the fraction to sample from
		data_train.

		target_column (string): the name of the target column in data_train


	Outputs:
		estimators (dictionary): a dictionary mapping the estimator number to the 
		scikit-learn model.
	'''

	estimators = {}

	for i in range(number_of_estimators):

		#sample from the data_train set
		data_sample = resample( data_train,
			stratify = data_train[target_column],
			n_samples = int(bootstrap_fraction * len(data_train.index)),
			replace = True)

		#get the features
		X = data_sample.drop(columns=[target_column])
		#get the targets
		y = data_sample[target_column]

		#print progress
		print("Training", i + 1 , "Out Of", number_of_estimators)

		#clone scikit-learn model
		m = clone(model)
		
		#train scikit-learn model and add to estimators dictionary
		estimators[i] = m.fit(X, y)

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
		can be a scikit-learn function like accuracy_score

		target_column (string): The name of the column that holds the targets.

	Outputs:
		game_matrix (numpy array): The completed game_matrix
	'''

	#create a zero matrix to start the game
	game_matrix = np.zeros((number_of_diracs, len(estimators)))

	#for every fold in the game data
	for fold in game_data['__Fold__'].unique():
		#get the fold data from the game_data
		fold_data = game_data.loc[ game_data['__Fold__'] == fold ]
		#get the features of the fold data
		X = fold_data.drop(columns = [target_column, '__Fold__'])
		#get the targets of the targets of the fold data
		y = fold_data[target_column]
		
		for estimator_index in estimators:
			#get the loss from predicting on the fold data using the estimator 
			#at index i in the estimators. this is the score for that estimator  
			game_matrix[fold, estimator_index] = (
				loss(y, estimators[estimator_index].predict(X)) )

	#return the game matrix
	return game_matrix


def game_matrix_build_parallel_helper(
	fold, 
	estimator_index, 
	loss, 
	X, 
	y, 
	estimators):
	'''
	This is a helper function that is used to parallelize the construction
	of a game matrix. The function calculates the loss between the targets
	and prediction for 1 estimator. This function is used to do the prediction
	and loss calculation in parallel.


	Inputs:
		fold (integer): The fold number that correlates to the X and y
		
		estimator_index (integer): The index of the estimator in the estimators obtained
		from estimator_training

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as accuracy_score

		X (pandas dataframe): The features of the data to predict on

		y (pandas dataframe): The targets of the data that will be used in prediction

		estimators (dictionary): The dictionary of estimators that was obtained from 
		estimator_training 

	Outputs:
		fold (integer): The fold number of the data that was passed in. This is used
		because multiprocessing does not guarantee the results return in the
		same order as entered. The fold is used as a marker to tell what 
		prediction_loss corresponds to what intputs

		estimator_index (integer): This has a similar effect as fold

		prediction_loss (float): The prediction error when the estimator at 
		estimator_index is used to predict on X.
	'''

	prediction_loss = loss(y, estimators[estimator_index].predict(X))
	return [ fold, estimator_index, prediction_loss ]


def game_matrix_build_parallel(
	game_data,
	number_of_diracs,
	estimators,
	loss,
	target_column,
	CORES):

	'''
	This is a parallel version of the game_matrix_build function. It builds
	the matrix using multiple cores and returns the completed game matrix.
	With multiple cores it will wun faster than game_matrix_build.

	Inputs:
		game_data (pandas dataframe): The data to build the game_matrix
		
		number_of_diracs (integer >= 1): The number of folds in the data 

		estimators (dictionary): The dictionary of estimators that was obtained from 
		estimator_training 

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as accuracy_score

		target_column (string): The name of the column that contains the targets.

		CORES (integer >= 1): The number of cores to run in parallelization

	Outputs:
		game_matrix (numpy array): The completed game_matrix
	'''

	#syntax sugar for constants
	FOLD = 0
	ESTIMATOR = 1
	LOSS = 2

	#construct an initial game matrix
	game_matrix = np.zeros((number_of_diracs, len(estimators)))

	parallel_arguments = []
	#for every fold in the game data
	for fold in game_data['__Fold__'].unique():
		#get the fold data from the game data
		fold_data = game_data.loc[game_data['__Fold__'] == fold]
		#get the features of the fold data
		X = fold_data.drop(columns = [target_column, '__Fold__'])
		#get the targets of the fold data
		y = fold_data[target_column]

		for estimator_index in estimators:
			#append the arguments that will be passed to the helper function
			#and processed in parallel. it is unpredictable in what order
			#the results arguments will be processed and in what order
			#the results will be. the fold and estimator_index are passed
			#as arguments and passed back as part of the result to keep
			#track of what results correspond to what arguments
			parallel_arguments.append(
				[fold, estimator_index, loss, X, y, estimators])

	parallel_results = []
	#define a multiprocess pool that uses CORES number of cores
	with Pool(processes = CORES) as pool:
	
		#get the results from the multiprocess. this function calls the
		#multiprocess starmap which calls game_matrix_build_parallel_helper
		#for every list in parallel_arguments. the result is returned as a
		#nested list where each sublist is the result from one sublist in 
		#parallel arguments.
	  	parallel_results = pool.starmap(
					game_matrix_build_parallel_helper, 
					parallel_arguments)

	#for every result in parallel_results
	for i in range( len(parallel_results) ):
		#get the fold for the current result
		results_fold = parallel_results[i][FOLD]
		#get the estimator for the current result
		results_estimator = parallel_results[i][ESTIMATOR]
		#get the loss for the current result
		results_loss = parallel_results[i][LOSS]

		#fill in the appropriate slot for the game matrix
		game_matrix[ results_fold, results_estimator ] = results_loss

	return game_matrix


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
		a scikit-learn function such as accuracy_score
		
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
	loss = accuracy_score,
	purification = False,
	target_column = '__Target__',
	detailed_game = True,
	CORES = 1):

	'''
	This function solves the game using the decision theoretic approach. 
	Purification is used if specified.
	
	
	Inputs:
		data_UQ (pandas dataframe): This is the data that was partitioned using train_UQ_split

		number_of_diracs (integer >= 1): The number of folds in the data

		estimators (dictionary): The dictionary of estimators that was obtained from 
		estimator_training

		loss (scikit learn loss function): The loss function to evaluate the estimator predictions. Can be 
		a scikit-learn function such as accuracy_score

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

		#initialize a list for the results from calling a parallel helper function
		parallel_results = []
		#call the parallel helper function in parellel
		with Pool(processes = CORES) as pool:
			#get the results from the multiprocess. this function calls the
			#multiprocess starmap which calls seperate_sort_dicracs_parallel_helper
			#for every list in parallel_arguments. the result is returned as a
			#nested list where each sublist is the result from one sublist in 
			#parallel arguments.
			parallel_results = pool.starmap(
						seperate_sort_dicracs_parallel_helper, 
						parallel_arguments)

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
		final_game_matrix = game_matrix_build(
					data_UQ,
					number_of_diracs,
					estimators,
					loss,
					target_column)

		final_game_value, final_PII_probabilities = PII_view_solver(final_game_matrix)

	return final_game_matrix, final_game_value, final_PII_probabilities


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
		a scikit-learn function such as accuracy_score

		counter (integer): An integer passed in to keep track of the progress in the 
		parallel computation.

		total (integer): The total number of times that decision_predict_parallel_helper
		will be called.

	Outputs:
		prediction (array like): The prediction of the estimator on the features multiplied
		by its probability in PII_probabilities
	'''

	print("Predicting", counter + 1, "Out of", total)

	prediction = estimators[estimator_index].predict(X) * PII_probabilities[estimator_index]	
	
	return prediction 


def decision_predict_parallel(
	data_test = None,
	estimators = None,
	PII_probabilities = None,
	loss = accuracy_score,
	target_column = '__Target__',
	CORES = 1):

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
		a scikit-learn function such as accuracy_score

		target_column (string): The name of the column that holds the target variable. 
		The default is __Target__

		CORES (integer >= 1): The number of cores to run in parallelization

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

	#get the results from the multiprocess. this function calls the
	#multiprocess starmap which calls seperate_sort_dicracs_parallel_helper
	#for every list in parallel_arguments. the result is returned as a
	#nested list where each sublist is the result from one sublist in 
	#parallel arguments.
	with Pool(processes = CORES) as pool:
		parallel_results = pool.starmap(
					decision_predict_parallel_helper, 
					parallel_arguments)

	#The final prediction is the sum of the results of the parallel helper
	#the loss is the loss function applied to the test targets and the 
	#final predictions
	final_prediction = parallel_results[0]
	for i in range( 1, len(parallel_results) ):
		final_prediction = final_prediction + parallel_results[i]

	final_prediction = [int( round(x) ) for x in final_prediction]

	test_loss = loss(y, final_prediction)

	return final_prediction, test_loss
