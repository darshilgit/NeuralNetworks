# Parikh, Darshil
# 1001-55-7968
# 2018-09-09
# Assignment-01-02
import numpy as np
# This module calculates the activation function


def calculate_activation_function(weight, bias, input_array, type='Positive Linear'):
	try:
		net_value = weight * input_array + bias
		if type == 'Sigmoid':
			activation = 1.0 / (1 + np.exp(-net_value))
		elif type == "Linear":
			activation = net_value
		elif type == "Hyperbolic Tangent":
			activation = np.tanh(net_value)
		elif type == "Positive Linear":
			t = (net_value > 0)
			activation = net_value * t
		return activation
	except Exception as e:
		print(str(e))