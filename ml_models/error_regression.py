
from model_utilities import *
import os
import psutil
import resource
from pysal.model import spreg
from pysal.lib import weights
from pysal.explore import esda
import libpysal
import spreg
from libpysal.weights import Queen, W
from scipy import stats
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
import math
import time
import random


def errorRegressionWithOriginalData(path_data_array, outputColumnIndex):
    m_start = get_process_memory()
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

    ts_start = time.time()
    neighbours, weight_values = get_weight_from_grid(num_rows, num_cols, total_cell)
    weight_mat = W(neighbours, weight_values)
    model_error = spreg.ML_Error(y_data, x_data, w=weight_mat)
    print(model_error.summary)
    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))



def errorRegressionWithRepartitionedData(path_data_array, path_group_index, path_cell_index, path_group_data, outputColumnIndex):
    m_start = get_process_memory()
    file_original_data = open(path_data_array, "rb")
    original_data_2d = np.load(file_original_data)
    num_rows, num_cols, num_attrs = original_data_2d.shape
    total_cell = num_rows*num_cols

    file_cell_group_ind = open(path_group_index, "rb")
    file_cell_ind = open(path_cell_index, "rb")
    file_cell_group_feature = open(path_group_data, "rb")

    cell_group_index = np.load(file_cell_group_ind)
    cell_index = np.load(file_cell_ind)
    cell_group_feature = np.load(file_cell_group_feature)

    y_data = cell_group_feature[:, outputColumnIndex]
    y_data = y_data.reshape((len(y_data), 1))
    x_data = np.delete(cell_group_feature, outputColumnIndex, 1)  # second argument for column number, third arguument for column delete

    ts_start = time.time()
    neighbours, weight_values = get_weight_from_repartitioned_cell(cell_group_index, cell_index, num_rows, num_cols)
    weight_mat_group = W(neighbours, weight_values)
    model_error = spreg.ML_Error(y_data, x_data, w=weight_mat)
    print(model_error.summary)
    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))
