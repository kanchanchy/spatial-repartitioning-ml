
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


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


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


def get_adjacency_dict_from_grid(num_rows, num_cols, total_cell):
    adj_dict = {}
    for i in range(total_cell):
        row = math.floor(i/num_cols)
        col = i%num_cols
        n_list = []
        if (col - 1) >= 0:
            n_list.append(row * num_cols + (col - 1))
        if (col + 1) < num_cols:
            n_list.append(row * num_cols + (col + 1))
        if (row - 1) >= 0:
            n_list.append((row - 1) * num_cols + col)
        if (row + 1) < num_rows:
            n_list.append((row + 1) * num_cols + col)
        adj_dict[i] = n_list
    return adj_dict


def get_weight_from_grid(num_rows, num_cols, total_cell):
    neighbours = dict()
    weights = dict()
    for i in range(total_cell):
        row = math.floor(i/num_cols)
        col = i%num_cols
        n_list = []
        w_list = []
        if (col - 1) >= 0:
            n_list.append(row * num_cols + (col - 1))
            w_list.append(1)
        if (col + 1) < num_cols:
            n_list.append(row * num_cols + (col + 1))
            w_list.append(1)
        if (row - 1) >= 0:
            n_list.append((row - 1) * num_cols + col)
            w_list.append(1)
        if (row + 1) < num_rows:
            n_list.append((row + 1) * num_cols + col)
            w_list.append(1)
        neighbours[i] = n_list
        weights[i] = w_list
    return neighbours, weights


def get_weight_from_repartitioned_cell(cell_group_index, cell_index, num_rows, num_cols):
    neighbours = dict()
    weights = dict()
    
    for k in range(len(cell_group_index)):
        row_start = cell_group_index[k][0]
        row_end = cell_group_index[k][1]
        col_start = cell_group_index[k][2]
        col_end = cell_group_index[k][3]
        n_list = []
        w_list = []
        
        if row_start > 0:
            for col in range(col_start, col_end + 1):
                if cell_index[row_start - 1][col] not in n_list:
                    n_list.append(cell_index[row_start - 1][col])
                    w_list.append(1)
        if row_end < num_rows - 1:
            for col in range(col_start, col_end + 1):
                if cell_index[row_end + 1][col] not in n_list:
                    n_list.append(cell_index[row_end + 1][col])
                    w_list.append(1)
        if col_start > 0:
            for row in range(row_start, row_end + 1):
                if cell_index[row][col_start - 1] not in n_list:
                    n_list.append(cell_index[row][col_start - 1])
                    w_list.append(1)
        if col_end < num_cols - 1:
            for row in range(row_start, row_end + 1):
                if cell_index[row][col_end + 1] not in n_list:
                    n_list.append(cell_index[row][col_end + 1])
                    w_list.append(1)
        
        neighbours[k] = n_list
        weights[k] = w_list
    return neighbours, weights


def get_cells_from_repartitioned_group(cell_group_index):
    group_cells = dict()
    
    for k in range(len(cell_group_index)):
        cell_list = []
        for i in range(cell_group_index[k][0], cell_group_index[k][1] + 1):
            for j in range(cell_group_index[k][2], cell_group_index[k][3] + 1):
                cell_list.append([i, j])
        group_cells[k] = cell_list
    return group_cells


def split_train_test_gwr(coords_x, coords_y, x_data, y_data, frac=0.2):
    removed_idx = np.random.randint(0, len(x_data)-1, size=int(frac * len(x_data)))
    x_test = x_data[removed_idx]
    y_test = y_data[removed_idx]
    coords_x_test = coords_x[removed_idx]
    coords_y_test = coords_y[removed_idx]
    x_train = np.delete(x_data, removed_idx, 0)
    y_train = np.delete(y_data, removed_idx, 0)
    coords_x_train = np.delete(coords_x, removed_idx, 0)
    coords_y_train = np.delete(coords_y, removed_idx, 0)
    
    coords_train = list(zip(coords_x_train, coords_y_train))
    coords_test = list(zip(coords_x_test, coords_y_test))
    return coords_train, coords_test, x_train, x_test, y_train, y_test


def create_train_test(dataset: np.array, frac=0.2):
    removed_idx = np.random.randint(0, len(dataset)-1, size=int(frac * len(dataset)))
    test_set = dataset[removed_idx]
    training_set = np.delete(dataset, removed_idx, 0)
    return training_set, test_set

