
from model_utilities import *
import os
import psutil
import resource
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from scipy import stats
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
import math
import time
import random
from sklearn.metrics import r2_score, mean_squared_error


def gwrRegressionWithOriginalData(path_data_array, path_centroid_array, outputColumnIndex):
    m_start = get_process_memory()
    file_centroid_data= open(path_centroid_array, "rb")
    centroid_data = np.load(file_centroid_data)
    centroid_points = np.delete(centroid_data, 2, 1)
    centroid_x, centroid_y = centroid_points.T

    file_original_data = open(path_data_array, "rb")
    original_data_2d = np.load(file_original_data)
    num_rows, num_cols, num_attrs = original_data_2d.shape
    total_cell = num_rows*num_cols

    original_data = np.zeros(shape = (num_rows*num_cols, num_attrs))
    for i in range(num_rows):
        for j in range(num_cols):
            original_data[i * num_cols + j] = original_data_2d[i][j]

    y_data = original_data[:, outputColumnIndex]
    y_data = y_data.reshape((len(y_data), 1))
    x_data = np.delete(original_data, outputColumnIndex, 1)  # second argument for column number, third arguument for column delete

    coords_train, coords_test, x_train, x_test, y_train, y_test = split_train_test_gwr(centroid_x, centroid_y, x_data, y_data, 0.2)

    ts_start = time.time()
    bw_optimum = Sel_BW(coords_train, y_train, x_train, kernel='gaussian', fixed = False).search(criterion='AICc')
    model_gwr = GWR(coords_train, y_train, x_train, bw=bw_optimum, kernel='gaussian', fixed=False)
    train_results = model_gwr.fit()
    aicc = train_results.aicc
    aic = train_results.aic
    ts_end = time.time()
    print("Bandwidth for bisquare: " + str(bw_optimum))
    print("AICc: " + str(aicc))
    print("AIC: " + str(aic))
    print("Time required to train the Model: " + str(ts_end - ts_start) + " seconds")

    prediction_result = model_gwr.predict(np.array(coords_test), np.array(x_test))
    test_predictions = prediction_result.predictions
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)
    print("Root mean square error: " + str(rmse))
    mae = abs(y_test - test_predictions)
    print('Mean Absolute Error:', round(np.mean(mae), 2), 'degrees.')

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))



def gwrRegressionWithRepartitionedData(path_group_data, path_group_centroid_array, outputColumnIndex):
    m_start = get_process_memory()
    file_group_centroid_data = open(path_group_centroid_array, "rb")
    group_centroid_data = np.load(file_group_centroid_data)
    centroid_points = np.delete(group_centroid_data, 2, 1)
    centroid_x, centroid_y = centroid_points.T

    file_cell_group_feature = open(path_group_data, "rb")
    cell_group_feature = np.load(file_cell_group_feature)

    y_data = cell_group_feature[:, outputColumnIndex]
    y_data = y_data.reshape((len(y_data), 1))
    x_data = np.delete(cell_group_feature, outputColumnIndex, 1)  # second argument for column number, third arguument for column delete

    coords_train, coords_test, x_train, x_test, y_train, y_test = split_train_test_gwr(centroid_x, centroid_y, x_data, y_data, 0.2)

    ts_start = time.time()
    bw_optimum = Sel_BW(coords_train, y_train, x_train, kernel='gaussian', fixed = False).search(criterion='AICc')
    model_gwr = GWR(coords_train, y_train, x_train, bw=bw_optimum, kernel='gaussian', fixed=False)
    train_results = model_gwr.fit()
    aicc = train_results.aicc
    aic = train_results.aic
    ts_end = time.time()
    print("Bandwidth for bisquare: " + str(bw_optimum))
    print("AICc: " + str(aicc))
    print("AIC: " + str(aic))
    print("Time required to train the Model: " + str(ts_end - ts_start) + " seconds")

    prediction_result = model_gwr.predict(np.array(coords_test), np.array(x_test))
    test_predictions = prediction_result.predictions
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)
    print("Root mean square error: " + str(rmse))
    mae = abs(y_test - test_predictions)
    print('Mean Absolute Error:', round(np.mean(mae), 2), 'degrees.')

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))
