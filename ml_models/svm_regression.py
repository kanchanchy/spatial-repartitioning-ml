
from model_utilities import *
import os
import psutil
import resource
from scipy import stats
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
import math
import time
import random
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def svmRegressionWithOriginalData(path_data_array, outputColumnIndex):
    m_start = get_process_memory()
    file_original_data = open(path_data_array, "rb")
    original_data_2d = np.load(file_original_data)
    num_rows, num_cols, num_attrs = original_data_2d.shape
    total_cell = num_rows*num_cols

    original_data = np.zeros(shape = (num_rows*num_cols, num_attrs))
    for i in range(num_rows):
        for j in range(num_cols):
            original_data[i * num_cols + j] = original_data_2d[i][j]

    labels = original_data[:, outputColumnIndex]
    features = np.delete(original_data, outputColumnIndex, 1)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)

    ts_start = time.time()
    svmRegressor = SVR(kernel='linear', C=15, gamma=0.5, epsilon=.01)
    svmRegressor.fit(train_features, train_labels)
    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    predictions = svmRegressor.predict(test_features)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    print("Root mean square error: " + str(rmse))
    ae = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(ae), 2), 'degrees.')

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))



def svmRegressionWithRepartitionedData(path_group_data, outputColumnIndex):
    m_start = get_process_memory()
    file_cell_group_feature = open(path_group_data, "rb")
    cell_group_feature = np.load(file_cell_group_feature)

    labels = cell_group_feature[:, outputColumnIndex]
    features = np.delete(cell_group_feature, outputColumnIndex, 1)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)

    ts_start = time.time()
    svmRegressor = SVR(kernel='linear', C=15, gamma=0.5, epsilon=.01)
    svmRegressor.fit(train_features, train_labels)
    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    predictions = svmRegressor.predict(test_features)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    print("Root mean square error: " + str(rmse))
    ae = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(ae), 2), 'degrees.')

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))
