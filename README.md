# A Machine Learning Aware Spatial Data Re-partitioning Framework for Spatial Datasets
This repository contains codes for the paper [A Machine Learning-Aware Data Re-partitioning Framework for Spatial Datasets](https://ieeexplore.ieee.org/document/9835487). This framework aims at reducing the training time and memory usage of a spatial machine learning model by reducing the number of partitions in a spatial grid dataset. Four types of datasets are used for experiments:
1. NYC Taxi Trip Multivariate Dataset
2. NYC Taxi Trip Univariate Dataset
3. Washington King County Home Sales Multivariate Dataset
4. Chicago Abandoned Cars Univariate Dataset

## Downloading Dataset
In order to experiment with NYC taxi trip dataset, download the 'Yellow Taxi Trip Records' for January 2009 from the site: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page. Put the downloaded CSV file inside this folder 'data/taxi_trip'. The file name should be 'yellow_tripdata_2009-01.csv'. Other datasets are available under 'data' folder.

## Data Preprocessing
Run data_preprocessing.py file. Main method contains four method calls for four datasets. In order to perform preprocessing on only one dataset, comment method calls for other datasets.

## Spatial Data Re-partitioning
Run repartitioning.py file. Main method contains re-partitioning steps for all four datasets. In order to perform re-partitioning on only one dataset, comment repartitioning steps for other datasets. In order to experiment different threshold of information loss, change the value of the variable infoLossThreshold_NYC_Multi .

## Training and Testing Machine Learning Models
Run train_test_models.py file. Sevel methods perform training and testing on seven types of machine learning models. In order to perform training and testing on only one model, comment method calls for other machine learning models. Regression models are tested with NYC taxi trip multivariate dataset (paths are defined by first set of global variables). Spatial kriging and clustering are performed on NYC taxi trip univariate dataset (paths are defined by second set of global variables). Change the paths to different dataset in order to perform training and testing on different dataset.

## BibTex for Citing the Work:
```
@INPROCEEDINGS{9835487,
  author={Chowdhury, Kanchan and Meduri, Venkata Vamsikrishna and Sarwat, Mohamed},
  booktitle={2022 IEEE 38th International Conference on Data Engineering (ICDE)}, 
  title={A Machine Learning-Aware Data Re-partitioning Framework for Spatial Datasets}, 
  year={2022},
  volume={},
  number={},
  pages={2426-2439},
  doi={10.1109/ICDE53745.2022.00227}
}
```

