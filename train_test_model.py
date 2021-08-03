
from ml_models.lag_regression import *
from ml_models.error_regression import *
from ml_models.gwr_regression import *
from ml_models.svm_regression import *
from ml_models.rf_regression import *
from ml_models.spatial_kriging import *
from ml_models.spatial_clustering import *


# Regression using NYC taxi trip multivariate dataset
outputColumnIndex = 3  # will be 0 for wa home sales dataset
path_data_array = "data/processed_data/nyc_multivariate_data/nyc_multivariate_grid.npy"
path_centroid_array = "data/processed_data/nyc_multivariate_data/centroid_data.npy"
path_group_index = "data/repartitioned_data/nyc_multivariate_data/cell_group_ind.npy"
path_cell_index = "data/repartitioned_data/nyc_multivariate_data/cell_ind.npy"
path_group_data = "data/repartitioned_data/nyc_multivariate_data/cell_group_feature.npy"
path_group_centroid_array = "data/repartitioned_data/nyc_multivariate_data/group_centroid_data.npy"

# Kriging and Clustering with NYC taxi trip univariate dataset
path_data_array_univariate = "data/processed_data/nyc_univariate_data/nyc_multivariate_grid.npy"
path_centroid_data_univariate = "data/processed_data/nyc_univariate_data/centroid_data.npy"
path_polygon_cells_univariate = "data/processed_data/nyc_univariate_data/polygon_cells.csv"
path_group_index_univariate = "data/repartitioned_data/nyc_univariate_data/cell_group_ind.npy"
path_cell_index_univariate = "data/repartitioned_data/nyc_univariate_data/cell_ind.npy"
path_group_data_univariate = "data/repartitioned_data/nyc_univariate_data/cell_group_feature.npy"
path_group_centroid_data_univariate = "data/repartitioned_data/nyc_univariate_data/group_centroid_data.npy"
path_polygon_groups_univariate = "data/repartitioned_data/nyc_univariate_data/polygon_groups.csv"


def train_test_lag_regression():
    # Train and test spatial lag regression with original data
    lagRegressionWithOriginalData(path_data_array, outputColumnIndex)
     # Train and test spatial lag regression with repartitioned data
    lagRegressionWithRepartitionedData(path_data_array, path_group_index, path_cell_index, path_group_data, outputColumnIndex)

def train_test_error_regression():
    # Train and test spatial error regression with original data
    errorRegressionWithOriginalData(path_data_array, outputColumnIndex)
    # Train and test spatial error regression with repartitioned data
    errorRegressionWithRepartitionedData(path_data_array, path_group_index, path_cell_index, path_group_data, outputColumnIndex)

def train_test_gwr_regression():
    # Train and test spatial GWR regression with original data
    gwrRegressionWithOriginalData(path_data_array, path_centroid_array, outputColumnIndex)
    # Train and test spatial GWR regression with repartitioned data
    gwrRegressionWithRepartitionedData(path_group_data, path_group_centroid_array, outputColumnIndex)

def train_test_svm_regression():
    # Train and test spatial SVM regression with original data
    svmRegressionWithOriginalData(path_data_array, outputColumnIndex)
    # Train and test spatial SVM regression with repartitioned data
    svmRegressionWithRepartitionedData(path_group_data, outputColumnIndex)

def train_test_rf_regression():
    # Train and test spatial random forest regression with original data
    rfRegressionWithOriginalData(path_data_array, outputColumnIndex)
    # Train and test spatial random forest regression with repartitioned data
    rfRegressionWithRepartitionedData(path_group_data, outputColumnIndex)

def train_test_spatial_kriging():
    # Train and test spatial krigingn with original data
    krigingWithOriginalData(path_centroid_data_univariate)
    # Train and test spatial kriging with repartitioned data
    krigingWithRepartitionedData(path_group_centroid_data_univariate)

def test_spatial_clustering():
    # Train and test spatial clustering with original data
    clusteringWithOriginalData(path_data_array_univariate, path_polygon_cells_univariate)
    # Train and test spatial clustering with repartitioned data
    clusteringWithRepartitionedData(path_data_array_univariate, path_group_index_univariate, path_cell_index_univariate, path_group_data_univariate, path_polygon_groups_univariate)


def main():
    train_test_lag_regression()
    train_test_error_regression()
    train_test_gwr_regression()
    train_test_svm_regression()
    train_test_rf_regression()
    train_test_spatial_kriging()
    test_spatial_clustering()


if __name__ == "__main__":
    main()
