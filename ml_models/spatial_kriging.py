
from model_utilities import *
import os
import psutil
import resource
import geopandas as gpd
from shapely import wkt
from pysal.model import spreg
from pysal.lib import weights
from pysal.explore import esda
import libpysal
import spreg
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from libpysal.examples import load_example
from libpysal.weights import Queen, W
from scipy import stats
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import seaborn
import math
import time
import random
from pyinterpolate.io_ops import read_point_data
from pyinterpolate.semivariance import calculate_semivariance  # experimental semivariogram
from pyinterpolate.semivariance import TheoreticalSemivariogram  # theoretical models
from pyinterpolate.kriging import Krige  # kriging models
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split


def krigingWithOriginalData(path_centroid_data):
    m_start = get_process_memory()

    file_centroid_data= open(path_centroid_data)
    centroid_data = np.load(file_centroid_data)

    train_set, test_set = create_train_test(centroid_data, frac=0.2)

    ts_start = time.time()
    search_radius = 0.01
    max_range = 0.32
    exp_semivar = calculate_semivariance(data=train_set, step_size=search_radius, max_range=max_range)
    semivar = TheoreticalSemivariogram(points_array=train_set, empirical_semivariance=exp_semivar)
    number_of_ranges = len(exp_semivar)  # The same number of ranges as experimental semivariogram
    semivar.find_optimal_model(weighted=False, number_of_ranges=number_of_ranges)
    model = Krige(semivariogram_model=semivar, known_points=train_set)

    neighbours = [4, 8, 16, 32]
    for nn in neighbours:
        mae, rmse = test_ordinary_kriging(kriging_model=model, test_values=test_set, number_of_neighbors=nn)
        print('Number of neighbors:', nn)
        print("Mean absolute error: " + str(mae))
        print("Root mean square error: " + str(rmse))
        print('\n')

    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))



def krigingWithRepartitionedData(path_group_centroid_data):
    m_start = get_process_memory()

    file_group_centroid_data= open(path_group_centroid_data, "rb")
    group_centroid_data = np.load(file_group_centroid_data)

    train_set, test_set = create_train_test(group_centroid_data, frac=0.2)

    ts_start = time.time()
    search_radius = 0.01
    max_range = 0.32
    exp_semivar = calculate_semivariance(data=train_set, step_size=search_radius, max_range=max_range)
    semivar = TheoreticalSemivariogram(points_array=train_set, empirical_semivariance=exp_semivar)
    number_of_ranges = len(exp_semivar)  # The same number of ranges as experimental semivariogram
    semivar.find_optimal_model(weighted=False, number_of_ranges=number_of_ranges)
    model = Krige(semivariogram_model=semivar, known_points=train_set)

    neighbours = [4, 8, 16, 32]
    for nn in neighbours:
        mae, rmse = test_ordinary_kriging(kriging_model=model, test_values=test_set, number_of_neighbors=nn)
        print('Number of neighbors:', nn)
        print("Mean absolute error: " + str(mae))
        print("Root mean square error: " + str(rmse))
        print('\n')

    ts_end = time.time()
    print("Time required to train the model: " + str(ts_end - ts_start) + " seconds")

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))
