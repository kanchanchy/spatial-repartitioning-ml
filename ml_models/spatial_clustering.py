
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
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering


def clusteringWithOriginalData(path_data_array, path_polygon_cells):
    m_start = get_process_memory()

    polyDf = pd.read_csv(path_polygon_cells)
    polyDf['geom'] = polyDf['geom'].apply(wkt.loads)
    polyGdf = gpd.GeoDataFrame(polyDf, geometry = polyDf.geom)

    file_original_data = open(path_data_array, "rb")
    original_data_2d = np.load(file_original_data)
    num_rows, num_cols, num_attrs = original_data_2d.shape
    total_cell = num_rows*num_cols

    original_data = np.zeros(shape = (num_rows*num_cols, num_attrs))
    for i in range(num_rows):
        for j in range(num_cols):
            original_data[i * num_cols + j] = original_data_2d[i][j]

    ts_start = time.time()
    weight_mat = get_weight_from_grid(num_rows, num_cols, total_cell)
    np.random.seed(123456)
    schc_model_cell = AgglomerativeClustering(linkage='ward', connectivity=weight_mat.sparse, n_clusters = 6)
    schc_model_cell.fit(original_data)
    cell_cluster_labels = schc_model_cell.labels_
    ts_end = time.time()
    print("Time required to cluster the model: " + str(ts_end - ts_start) + " seconds")

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))

    polyGdf['clusters'] = cell_cluster_labels
    f, ax = plt.subplots(1, figsize=(9, 9))
    polyGdf.plot(column='clusters', categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Spatial Clusters')
    plt.show()



def clusteringWithRepartitionedData(path_data_array, path_group_index, path_cell_index, path_group_data, path_polygon_groups):
    m_start = get_process_memory()

    polyGroupDf = pd.read_csv(path_polygon_groups)
    polyGroupDf['geom'] = polyGroupDf['geom'].apply(wkt.loads)
    polyGroupGDf = gpd.GeoDataFrame(polyGroupDf, geometry = polyGroupDf.geom)

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

    ts_start = time.time()
    weight_mat_group = get_weight_from_repartitioned_cell(cell_group_index, cell_index, num_rows, num_cols)
    np.random.seed(123456)
    schc_model_group = AgglomerativeClustering(linkage='ward', connectivity=weight_mat_group.sparse, n_clusters = 6)
    schc_model_group.fit(cell_group_feature)
    group_cluster_labels = schc_model_group.labels_
    ts_end = time.time()
    print("Time required to cluster the model: " + str(ts_end - ts_start) + " seconds")

    m_end = get_process_memory()
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("Maximum memory usage: ", str(max_memory))
    print("Memory usage at start, end, and consumption: ", str(m_start), str(m_end), str(m_end - m_start))

    polyGroupGDf['clusters'] = group_cluster_labels
    f, ax = plt.subplots(1, figsize=(9, 9))
    polyGroupGDf.plot(column='clusters', categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Spatial Group Clusters')
    plt.show()
