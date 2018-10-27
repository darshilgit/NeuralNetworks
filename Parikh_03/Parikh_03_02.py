# Parikh, Darshil
# 1001-55-7968
# 2018-10-08
# Assignment-03-02

import numpy as np
from sklearn.metrics import confusion_matrix
#from random import shuffle

target_vector = []
conf_matrix = []
# confusion_matrix = np.zeros((10, 10))
# This module calculates the activation function


def calculate_activation(net_value, type):
	activation = None
	if type == "Linear":
		activation = net_value
		max_index = np.argmax(net_value, axis=1)
		for i in range(len(max_index)):
			activation[i] = np.zeros(10)
			activation[i][max_index[i]] = 1
	elif type == "Hyperbolic Tangent":
		activation = np.tanh(net_value)
		max_index = np.argmax(net_value, axis=1)
		for i in range(len(max_index)):
			activation[i] = np.zeros(10)
			activation[i][max_index[i]] = 1
	elif type == "Symmetrical Hard Limit":
		net_value[net_value >= 0] = 1
		net_value[net_value < 0] = 0
		activation = net_value
	return activation


def prediction(net_value, target, activation_function):

	if activation_function == 'Linear':
		actual = calculate_activation(net_value, activation_function)
		error_rate = (calculate_error(target, actual)/200) * 100
	elif activation_function == 'Hyperbolic Tangent':
		actual = calculate_activation(net_value, activation_function)
		error_rate = (calculate_error(target, actual)/200) * 100
	elif activation_function == 'Symmetrical Hard Limit':
		actual = calculate_activation(net_value, activation_function)
		error_rate = (calculate_error(target, actual)/200) * 100
	return error_rate


def calculate_error(target, actual):
	error_count = 0
	#if activation_function == 'Linear' or activation_function == 'Hyperbolic Tangent':
	for i in range(len(target)):
		if not(np.array_equal(target[i],actual[i])):
			error_count += 1

	global conf_matrix
	conf_matrix = np.zeros((10,10))
	conf_matrix = confusion_matrix((np.array(target)).argmax(axis=1), actual.argmax(axis=1))
	#print(conf_matrix)
	# else:
	# 	for i in range(len(target)):
	# 		if not (np.array_equal(target, actual)):
	# 			error_count += 1
	return error_count


def filtered_learning(target, input, alpha, old_weight):
	first_term = (1 - alpha) * old_weight
	second_term = (np.array(input).transpose().dot(target)) * alpha
	new_weight = first_term + second_term
	return new_weight


def delta_rule_learning(target, input, alpha, old_weight, actual):
	first_term = old_weight
	second_term = (np.array(input).transpose().dot(target-actual)) * alpha
	new_weight = first_term + second_term
	return new_weight


def unsupervised_learning(input, alpha, old_weight, actual):
	first_term = old_weight
	second_term = (np.array(input).transpose().dot(actual)) * alpha
	new_weight = first_term + second_term
	return new_weight


def get_target_vector(input_vector):
	target_vector = []
	a = np.array(input_vector)[:, [785]]
	for ele in a:
		dummy_vector = np.zeros(10)
		dummy_vector[int(ele[0])] = 1
		target_vector.append(dummy_vector)
	return target_vector


def calculate_activation_function(weight, alpha, input_vector, learning_method, type, global_epoch):
	# confusion_matrix = np.zeros((10, 10))
	#conf_matrix = []
	error_rate = []
	epochs = []
	#shuffle(input_vector)
	training_input_vector = input_vector[:800]
	testing_input_vector = input_vector[800:]
	train_target = get_target_vector(training_input_vector)
	training_input_vector = (np.array(training_input_vector))[:, :-1]
	test_target = get_target_vector(testing_input_vector)
	testing_input_vector = (np.array(testing_input_vector))[:, :-1]
	epoch = 100
	while epoch > 0:
		if learning_method == 'Filtered Learning':
			weight = filtered_learning(train_target, training_input_vector, alpha, weight)
			net_value_testing = testing_input_vector.dot(weight)
			error_rate.append(prediction(net_value_testing, test_target, type))
			epochs.append(101 - epoch + global_epoch)
			epoch -= 1
		else:
			net_value = np.dot(training_input_vector,weight)
			activation = calculate_activation(net_value, type)
			if learning_method == 'Delta Rule':
				weight = delta_rule_learning(train_target, training_input_vector, alpha, weight, activation)
				net_value_testing = testing_input_vector.dot(weight)
				# print(net_value_testing)
				error_rate.append(prediction(net_value_testing, test_target, type))
				epochs.append(101-epoch + global_epoch)
				epoch -= 1
			elif learning_method == 'Unsupervised Hebb':
				weight = unsupervised_learning(training_input_vector, alpha, weight, activation)
				net_value_testing = testing_input_vector.dot(weight)
				error_rate.append(prediction(net_value_testing, test_target, type))
				epochs.append(101 - epoch + global_epoch)
				epoch -= 1
	#print(weight)
	return error_rate, weight, 100, epochs, conf_matrix