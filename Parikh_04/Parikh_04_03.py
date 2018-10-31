# Parikh, Darshil
# 1001-55-7968
# 2018-10-28
# Assignment-04-03

import numpy as np

#file_name = "data_set_1.csv"
maximum = 1
minimum = -1


def read_csv_as_matrix(file_name):
    # Each row of data in the file becomes a row in the matrix
    # So the resulting matrix has dimension [num_samples x sample_dimension]
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
    price_scaled, volume_scaled = normalize_data(data)
    return price_scaled, volume_scaled



def normalize_data(data):
    price_change= np.array(data[:, 0])
    price_change_std = (price_change - price_change.min(axis=0)) / (price_change.max(axis=0) - price_change.min(axis=0))
    price_scaled = price_change_std * (maximum- minimum) + minimum

    volume_change = np.array(data[:, 1])
    volume_change_std = (volume_change - volume_change.min(axis=0)) / (volume_change.max(axis=0) - volume_change.min(axis=0))
    volume_scaled = volume_change_std * (maximum - minimum) + minimum

    #print(np.where(volume_scaled == volume_scaled.min()))
    return price_scaled, volume_scaled
