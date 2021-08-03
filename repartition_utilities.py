
from pyspark.sql import SparkSession
from pyspark import StorageLevel
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import random
from datetime import datetime
import time
import resource
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import LongType
from pyspark.sql.types import IntegerType
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from sedona.register import SedonaRegistrator
from sedona.core.SpatialRDD import SpatialRDD
from sedona.core.SpatialRDD import PointRDD
from sedona.core.SpatialRDD import PolygonRDD
from sedona.core.SpatialRDD import LineStringRDD
from sedona.core.enums import FileDataSplitter
from sedona.utils.adapter import Adapter
from sedona.core.spatialOperator import KNNQuery
from sedona.core.spatialOperator import JoinQuery
from sedona.core.spatialOperator import JoinQueryRaw
from sedona.core.spatialOperator import RangeQuery
from sedona.core.spatialOperator import RangeQueryRaw
from sedona.core.formatMapper.shapefileParser import ShapefileReader
from sedona.core.formatMapper import WkbReader
from sedona.core.formatMapper import WktReader
from sedona.core.formatMapper import GeoJsonReader
from sedona.sql.types import GeometryType
from sedona.core.SpatialRDD import RectangleRDD
from sedona.core.geom.envelope import Envelope
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from sedona.core.formatMapper.shapefileParser import ShapefileReader
from sedona.core.enums import GridType
from sedona.core.enums import IndexType
from pyspark.sql.functions import monotonically_increasing_id, when, col


DATA_TYPE_INT = 0
DATA_TYPE_FLOAT_DOUBLE = 1


def get_adjacency_from_grid(num_rows, num_cols, total_cell):
    adj_matrix = np.zeros(shape = (total_cell, total_cell))
    for i in range(total_cell):
        row = math.floor(i/num_cols)
        col = i%num_cols
        if (col - 1) >= 0:
            adj_matrix[i][row * num_cols + (col - 1)] = 1
        if (col + 1) < num_cols:
            adj_matrix[i][row * num_cols + (col + 1)] = 1
        if (row - 1) >= 0:
            adj_matrix[i][(row - 1) * num_cols + col] = 1
        if (row + 1) < num_rows:
            adj_matrix[i][(row + 1) * num_cols + col] = 1
    return adj_matrix


def get_cells_from_repartitioned_group(cell_group_index):
    group_cells = dict()
    
    for k in range(len(cell_group_index)):
        cell_list = []
        for i in range(cell_group_index[k][0], cell_group_index[k][1] + 1):
            for j in range(cell_group_index[k][2], cell_group_index[k][3] + 1):
                cell_list.append([i, j])
        group_cells[k] = cell_list
        
    return group_cells


def findMinAttrVariation(data_attribute, current_min_diff):
    minDiff = np.max(data_attribute)
    for i in range(len(data_attribute)-1):
        for j in range(len(data_attribute[i])-1):
            sumDiff1 = 0
            sumDiff2 = 0
            for k in range(len(data_attribute[i][j])):
                sumDiff1 += abs(data_attribute[i][j][k] - data_attribute[i][j+1][k])
                sumDiff2 += abs(data_attribute[i][j][k] - data_attribute[i+1][j][k])
            if sumDiff1 < minDiff and sumDiff1 > current_min_diff:
                minDiff = sumDiff1
            if sumDiff2 < minDiff and sumDiff2 > current_min_diff:
                minDiff = sumDiff2
    return minDiff


def findMinVariationGroups(data_attribute, minDiff):
    cell_group_index = []
    cell_index = np.zeros(shape = (len(data_attribute), len(data_attribute[0])), dtype = int)
    visited = np.zeros(shape = (len(data_attribute), len(data_attribute[0])))
    
    i = 0
    j = 0
    while i < len(data_attribute):
        if visited[i][j] != 0:
            if j < len(data_attribute[0]) - 1:
                j += 1
            else:
                i += 1
                j = 0
            continue
            
        col = 0
        while (j+col) < len(data_attribute[0]) - 1:
            sumDiff = 0
            for k in range(len(data_attribute[0][0])):
                sumDiff += abs(data_attribute[i][j+col+1][k] - data_attribute[i][j+col][k])
            if sumDiff <= minDiff:
                col += 1
            else:
                break
        row = 0
        while (i+row) < len(data_attribute) - 1:
            for k in range(len(data_attribute[0][0])):
                sumDiff += abs(data_attribute[i+row+1][j][k] - data_attribute[i+row][j][k])
            if sumDiff <= minDiff:
                row += 1
            else:
                break
        
        rec_hori = 0
        if row > 0:
            rec_hori = col
            for n in range(j + 1, j + col + 1):
                for m in range(i, i + row + 1):
                    for k in range(len(data_attribute[0][0])):
                        sumDiff += abs(data_attribute[m][n][k] - data_attribute[m][n-1][k])
                    if sumDiff > minDiff:
                        rec_hori = n - j -1
                        break
                if rec_hori < col:
                    break
        rec_vert = 0
        if col > 0:
            rec_vert = row
            for m in range(i + 1, i + row + 1):
                for n in range(j, j + col + 1):
                    for k in range(len(data_attribute[0][0])):
                        sumDiff += abs(data_attribute[m][n][k] - data_attribute[m-1][n][k])
                    if sumDiff > minDiff:
                        rec_vert = m - i -1
                        break
                if rec_vert < row:
                    break
                    
        total_cell = col
        col_inc = col
        row_inc = 0
        if row > total_cell:
            total_cell = row
            row_inc = row
            col_inc = 0
        if row * rec_hori >= total_cell:
            total_cell = row * rec_hori
            row_inc = row
            col_inc = rec_hori
        if col * rec_vert >= total_cell:
            total_cell = col * rec_vert
            row_inc = rec_vert
            col_inc = col
        
        for m in range(i, i + row_inc + 1):
            for n in range(j, j + col_inc + 1):
                visited[m][n] = 1
                cell_index[m][n] = len(cell_group_index)
        
        cell_group_index.append([i, i + row_inc, j, j + col_inc])
        if j + col_inc < len(data_attribute[0]) - 1:
            j += col_inc + 1
        else:
            i += 1
            j = 0
    return cell_group_index, cell_index


def assignFeatureToGroup(cell_group_index, data_attribute, data_types):
    cell_group_feature = [[0] * len(data_attribute[0][0]) for _ in range(len(cell_group_index))]
    for t in range(len(cell_group_index)):
        for k in range(len(data_attribute[0][0])):
            count = 0
            attr_sum = 0
            for i in range(cell_group_index[t][0], cell_group_index[t][1] + 1):
                for j in range(cell_group_index[t][2], cell_group_index[t][3] + 1):
                    attr_sum += data_attribute[i][j][k]
                    count += 1
            if data_types[k] == DATA_TYPE_FLOAT_DOUBLE:
                cell_group_feature[t][k] = round(attr_sum/count, 2)
            else:
                cell_group_feature[t][k] = round(attr_sum/count)
    return cell_group_feature


def calculateInfoLoss(original_data, new_data, cell_index):
    loss_total = 0
    for i in range(len(original_data)):
        for j in range(len(original_data[0])):
            temp_loss = 0
            for k in range(len(original_data[0][0])):
                divisor = original_data[i][j][k]
                if divisor == 0:
                    divisor = new_data[cell_index[i][j]][k]
                if divisor != 0:
                    temp_loss += abs(original_data[i][j][k] - new_data[cell_index[i][j]][k])/divisor
            loss_total += temp_loss/len(original_data[0][0])
    infoLoss = loss_total/(len(original_data)*len(original_data[0]))
    return infoLoss
    


def doRepartitioning(attrData, data_types, lossThreshold):
    attrData2 = attrData.reshape((len(attrData) * len(attrData[0]), len(data_types)))
    attrData2Norm = attrData2/attrData2.max(axis = 0)
    attrDataNorm = attrData2Norm.reshape((len(attrData), len(attrData[0]), len(data_types)))
    
    currentTotalLoss = 0
    prevTotalLoss = 0
    prev_cell_group_index = np.NaN
    prev_cell_index = np.NaN
    prev_cell_group_feature = np.NaN
    currentMinDiff = -1
    i = 0
    while True:
        i += 1
        min_variance = findMinAttrVariation(attrDataNorm, currentMinDiff)
        cell_group_index, cell_index = findMinVariationGroups(attrDataNorm, min_variance)
        cell_group_feature = assignFeatureToGroup(cell_group_index, attrData, data_types)
        infoLoss = calculateInfoLoss(attrData, cell_group_feature, cell_index)
        currentTotalLoss = infoLoss
        
        if currentTotalLoss <= lossThreshold:
            prevTotalLoss = currentTotalLoss
            prev_cell_group_index = cell_group_index
            prev_cell_index = cell_index
            prev_cell_group_feature = cell_group_feature
            currentMinDiff = min_variance
        else:
            return prevTotalLoss, prev_cell_group_index, prev_cell_index, prev_cell_group_feature 



# This method needs to be called to start repartitioning
def callRepartitioning(grid_data, attr_data_types, infoLossThreshold, outputPath):
    # Doing repartitioning
    print("Repartitioning started")
    ts_start = time.time()
    totalLoss, cell_group_index, cell_index, cell_group_feature = doRepartitioningMultiAttr(grid_data, attr_data_types, infoLossThreshold)
    ts_end = time.time()
    print("Repartitioning completed")
    print("Original Cell Count: ", str(len(grid_data) * len(grid_data[0])))
    print("New Cell Count: ", str(len(cell_group_feature)))
    print("Information Loss: ", str(totalLoss))
    print("Elapsed Time: " + str(ts_end - ts_start) + " Seconds")

    # Saving into files as numpy arrays
    file_cell_gr_index = open(outputPath + "/cell_group_ind.npy", "wb")
    file_cell_index = open(outputPath + "/cell_ind.npy", "wb")
    file_cell_gr_feature = open(outputPath + "/cell_group_feature.npy", "wb")
    np.save(file_cell_gr_index, cell_group_index)
    np.save(file_cell_index, cell_index)
    np.save(file_cell_gr_feature, cell_group_feature)

    # Extracting polygons of cell-groups
    print("Extracting polygons of cell-groups")
    group_cells = get_cells_from_repartitioned_group(cell_group_index)
    group_polies = []
    for k in range(len(cell_group_index)):
        cell_list = group_cells[k]
        cell_poly_list = []
        for cell in cell_list:
            cell_poly_list.append(grid_geom_list[cell[0]*num_cols + cell[1]])
        group_polies.append(gpd.GeoSeries(cascaded_union(cell_poly_list))[0])

    # Saving polygon groups into files
    polyGroupDf = pd.DataFrame(group_polies, columns=['geom'])
    polyGroupDf.to_csv(outputPath + "/polygon_groups.csv")
    print("Cell-groups extraction completed")

    # Calculating centroids of polygons
    print("Calculating centroids of cell-groups")
    group_centroids = list(map(lambda x: x.centroid, group_polies))
    group_centroids_list = list(map(lambda center_point: [center_point.x, center_point.y], group_centroids))

    group_centroid_data =  []
    for k in range(len(group_centroids_list)):
        group_centroid_data.append([group_centroids_list[k][0], group_centroids_list[k][1], cell_group_feature[k][0]])
    group_centroid_data = np.array(group_centroid_data)

    # Saving group centroids into files
    file_group_centroid_data = open("data/taxi_trip/part_36000/single_attribute/np_arrays/loss_0.15/group_centroid_data.npy", "wb")
    np.save(file_group_centroid_data, group_centroid_data)
    print("Cell-groups centroids calculation completed")





