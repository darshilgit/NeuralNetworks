# Parikh, Darshil
# 1001-55-7968
# 2018-09-24
# Assignment-02-02
import numpy as np
# This module calculates the activation function


def calculate_activation_function(weight1, weight2, bias, input_array, type='Symmetrical Hard Limit'):
	target = np.array([[1, 1, -1, -1]])
	epochs = 100
	activation = None
	weight = np.array([[weight1, weight2, bias]])
	for i in range(epochs):
		net_value = weight.dot(input_array)
		if type == "Linear":
			activation = net_value
		elif type == "Hyperbolic Tangent":
			activation = np.tanh(net_value)
		elif type == "Symmetrical Hard Limit":
			net_value[net_value >= 0] = 1
			net_value[net_value < 0] = -1
			activation = net_value
		i = 0
		while i < 4:
			error = target[0][i] - activation[0][i]
			# The following lines to handle NaN condition in Linear activation function
			# has been reference from the below mentioned link:
			#
			# https://stackoverflow.com/questions/48862451/perceptron-learning-algorithm-doesnt-work/48939759?noredirect=1#comment84890707_48939759
			if error > 1000 or error < -700:
				error /= 10000
			weight = weight + error * (input_array[:, [i]]).transpose()
			i += 1
		if np.array_equal(activation, target):
			break
	return weight[0][0], weight[0][1], weight[0][2]
