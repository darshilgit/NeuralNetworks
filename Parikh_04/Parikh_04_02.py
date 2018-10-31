# Parikh, Darshil
# 1001-55-7968
# 2018-10-28
# Assignment-04-02

import numpy as np
#from random import shuffle

# This module calculates the activation function


# def LMS_learning(weight, input, target, alpha):
#     training_net_value = np.dot(weight, (np.array(input)).transpose())
#     error = target - training_net_value
#     weight = weight + (2 * (alpha * np.array(error).dot(input)))
#     return weight


def calculate_activation_function(weight, alpha, input_values, target_values, training_sample_size, iterations, learn_type):
    # error = None
    mse = []
    mae = []
    training_input = input_values[:int(((training_sample_size/100)*len(input_values)))]
    testing_input = input_values[int(((training_sample_size/100)*len(input_values))):]
    training_target = target_values[:int(((training_sample_size/100)*len(input_values)))]
    testing_target = target_values[int(((training_sample_size/100)*len(input_values))):]

    training_input = np.array(training_input)
    training_target = np.array(training_target)
    testing_input = np.array(testing_input)
    testing_target = np.array(testing_target)

    # slides example
    # weight = np.array(np.zeros((1, 3)))
    # target = [-1, 1]
    # training_target = np.array(target)
    # alpha = 0.2
    if learn_type == 0:
        while iterations > 0:
            # training_input = np.array([[-1, 1, -1], [1, 1, -1]])
            # iterative approach
            for i in range(training_input.shape[0]):
                net_value = training_input[i].dot(weight)
                error = training_target[i] - net_value
                weight += ((2 * alpha) * (training_input[i].transpose().dot(error)))
            testing_actual = np.array(testing_input).dot(weight)
            error = testing_target - testing_actual
            mae.append(np.max(np.absolute(error)))
            mse.append(np.mean(np.square(error)))
            iterations -= 1

            # taking all inputs together in matrix
            # weight = LMS_learning(weight, training_input, training_target, alpha)
            # testing_actual = np.dot(weight, testing_input.transpose())
            # error = testing_target - testing_actual
            # squared_error = np.square(error)
            # mean_squared_error = np.mean(squared_error)
            # mse.append(mean_squared_error)
            # mean_absolute_error = np.max(error)
            # mae.append(mean_absolute_error)
            # iterations -= 1
        return mse, mae
    elif learn_type == 1:
        while iterations > 0:
            # print(training_input.shape[0])
            training_input_transpose = training_input.transpose()
            training_input_dot_training_input_transpose = training_input_transpose.dot(training_input)
            r_inv = np.linalg.inv(training_input_dot_training_input_transpose/training_input.shape[0])
            h = training_target.dot(training_input)/training_input.shape[0]
            weight = r_inv.dot(h)
            actual_res = testing_input.dot(weight)
            error = testing_target - actual_res
            squared_error = np.square(error)
            mean_squared_error = np.mean(squared_error)
            mse.append(mean_squared_error)
            mean_absolute_error = np.max(np.absolute(error))
            mae.append(mean_absolute_error)
            iterations -= 1
        return mse, mae


