
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


def knnClassificationWithOriginalData(path_data_array, outputColumnIndex):
    m_start = get_process_memory()
    file_original_data = open(path_data_array, "rb")
    original_data_2d = np.load(file_original_data)
    num_rows, num_cols, num_attrs = original_data_2d.shape
    total_cell = num_rows*num_cols

    original_data = np.zeros(shape = (num_rows*num_cols, num_attrs))
    for i in range(num_rows):
        for j in range(num_cols):
            original_data[i * num_cols + j] = original_data_2d[i][j]

    targets = original_data[:, outputColumnIndex]
    features = np.delete(original_data, outputColumnIndex, 1)

    low = np.percentile(targets, 20)
    low_mid = np.percentile(targets, 40)
    mid = np.percentile(targets, 60)
    mid_high = np.percentile(targets, 80)
    high = np.percentile(targets, 100)

    labels = np.array([0]*len(targets))
    for i in range(len(targets)):
        if targets[i] <= low:
            labels[i] = 0
        elif targets[i] <= low_mid:
            labels[i] = 1
        elif targets[i] <= mid:
            labels[i] = 2
        elif targets[i] <= mid_high:
            labels[i] = 3
        else:
            labels[i] = 4

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)

    ts_start = time.time()
    model = KNeighborsClassifier(n_neighbors = 6)
    model.fit(train_features, train_labels)
    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    pred_labels_te = model.predict(test_features)
    score_te = model.score(test_features, test_labels)
    print('Accuracy Score: ', score_te)
    print(classification_report(test_labels, pred_labels_te))

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))



def knnClassificationWithRepartitionedData(path_data_array, path_group_data, outputColumnIndex):
    m_start = get_process_memory()
    file_original_data = open(path_data_array, "rb")
    original_data_2d = np.load(file_original_data)
    num_rows, num_cols, num_attrs = original_data_2d.shape
    total_cell = num_rows*num_cols

    original_data = np.zeros(shape = (num_rows*num_cols, num_attrs))
    for i in range(num_rows):
        for j in range(num_cols):
            original_data[i * num_cols + j] = original_data_2d[i][j]

    file_cell_group_feature = open(path_group_data, "rb")
    cell_group_feature = np.load(file_cell_group_feature)

    org_targets = original_data[:, outputColumnIndex]
    targets = cell_group_feature[:, outputColumnIndex]
    features = np.delete(cell_group_feature, outputColumnIndex, 1)

    low = np.percentile(org_targets, 20)
    low_mid = np.percentile(org_targets, 40)
    mid = np.percentile(org_targets, 60)
    mid_high = np.percentile(org_targets, 80)
    high = np.percentile(org_targets, 100)

    labels = np.array([0]*len(targets))
    for i in range(len(targets)):
        if targets[i] <= low:
            labels[i] = 0
        elif targets[i] <= low_mid:
            labels[i] = 1
        elif targets[i] <= mid:
            labels[i] = 2
        elif targets[i] <= mid_high:
            labels[i] = 3
        else:
            labels[i] = 4

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)

    ts_start = time.time()
    model = KNeighborsClassifier(n_neighbors = 6)
    model.fit(train_features, train_labels)
    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    pred_labels_te = model.predict(test_features)
    score_te = model.score(test_features, test_labels)
    print('Accuracy Score: ', score_te)
    print(classification_report(test_labels, pred_labels_te))

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))

