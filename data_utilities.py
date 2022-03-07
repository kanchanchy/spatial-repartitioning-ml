
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
from pyproj import Proj, transform



spark = SparkSession.builder.master("local[*]").appName("Processing Spatial Repartitioning Data").config("spark.serializer", KryoSerializer.getName).config("spark.kryo.registrator", SedonaKryoRegistrator.getName).config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-2.4_2.11:1.0.0-incubating,org.datasyslab:geotools-wrapper:geotools-24.0").getOrCreate()

SedonaRegistrator.registerAll(spark)

sc = spark.sparkContext


def load_dataset(path_to_dataset, is_shape_file):
    if is_shape_file:
        obj = gpd.read_file(path_to_dataset)
        return obj

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

def row_normalize(adj_matrix):
        sum_of_adj_rows = adj_matrix.sum(axis=1)
        adj_matrix_norm = adj_matrix / sum_of_adj_rows[:, np.newaxis]
        adj_matrix_norm = np.nan_to_num(adj_matrix_norm, copy = False, nan = 0)
        return adj_matrix_norm

def partition_by_grid(dataset, x_interval, y_interval):
    xmin, ymin, xmax, ymax = dataset.total_bounds
    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), x_interval))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), y_interval))
    rows.reverse()
        
    polygons = []
    for y in rows:
        for x in cols:
            polygons.append(Polygon([(x, y), (x + x_interval, y), (x + x_interval, y - y_interval), (x, y - y_interval)]))
    return len(rows), len(cols), polygons

def partition_by_grid_nonshape_file(xmin, xmax, ymin, ymax, x_interval, y_interval):
    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), x_interval))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), y_interval))
    rows.reverse()
        
    polygons = []
    for y in rows:
        for x in cols:
            polygons.append(Polygon([(x, y), (x + x_interval, y), (x + x_interval, y - y_interval), (x, y - y_interval)]))
    return len(rows), len(cols), polygons



def process_taxi_trip_multivariate_data(lat_interval, lon_interval):
	# Loading shape file containing spatial information
	gdf = load_dataset("data/taxi_trip/taxi_zones/taxi_zones.shp", True)
	print("Shape file loaded")

	# Creating grid partitions
	num_rows, num_cols, grid_geom_list = partition_by_grid(gdf, lat_interval, lon_interval)

	polyListDf = pd.DataFrame(grid_geom_list, columns=['geom'])
	polyListDf.to_csv("data/processed_data/taxi_trip_multivariate_data/polygon_cells.csv")
	print("Grid partitions are created")

	# Defining schema of polygons
	schema = StructType(
		[
		StructField("_id", IntegerType(), False),
		StructField("geom", GeometryType(), False)
		]
		)

	# Creating list of polygons
	poly_data = []
	for i in range(len(grid_geom_list)):
		poly_data.append([i, grid_geom_list[i]])

	# Creating dataframes of polygons
	polyDf = spark.createDataFrame(poly_data, schema)
	polyDf.show(5, False)
	print("Polygon dataframe created")

	# Converting polygon dataframes into polygon RDDs
	polyRDD = Adapter.toSpatialRdd(polyDf, "geom")
	polyRDD.CRSTransform("epsg:2263", "epsg:2263")
	polyRDD.analyze()
	polyRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("Polygon dataframe converted into polygon RDD")


	# Loading CSV dataset
	tripDf = spark.read.format("csv").option("delimiter",",").option("header","true").load("data/taxi_trip/yellow_tripdata_2009-01.csv")
	tripDf = tripDf.withColumn("Serial_ID", monotonically_increasing_id())
	tripDf.createOrReplaceTempView("tripDf")
	tripDf.show(5, False)
	print("CSV dataset loaded")


	# Generating dataframe consisting of pickup information of selected attributes
	buildOnSpatialPartitionedRDD = True
	usingIndex = True
	considerBoundaryIntersection = True

	pickupInfoDf = spark.createDataFrame([], StructType([]))
	for i in range(15):
		start_id = i * 1000000
		end_id = (i + 1) * 1000000 - 1
		pointDf = spark.sql("select ST_Point(double(tripDf.Start_Lat), double(tripDf.Start_Lon)) as point_loc, int(tripDf.Passenger_Count) as passenger_count, float(tripDf.Trip_Distance) as trip_dist, float(tripDf.Fare_Amt) as fare from tripDf where tripDf.Serial_ID >= {0} and tripDf.Serial_ID <= {1}".format(start_id, end_id))
		pointRDD = Adapter.toSpatialRdd(pointDf, "point_loc")
		pointRDD.CRSTransform("epsg:4326", "epsg:2263")

		pointRDD.analyze()
		pointRDD.spatialPartitioning(polyRDD.getPartitioner())
		pointRDD.buildIndex(IndexType.QUADTREE, buildOnSpatialPartitionedRDD)
		result_pair_rdd = JoinQueryRaw.SpatialJoinQueryFlat(pointRDD, polyRDD, usingIndex, considerBoundaryIntersection)
		pickupInfoPartDf = Adapter.toDf(result_pair_rdd, polyRDD.fieldNames, pointRDD.fieldNames, spark)
		pickupInfoPartDf.createOrReplaceTempView("pickupInfoPartDf")

		pickupInfoPartDf = spark.sql("SELECT int(a._id) as _id, count(a.rightgeometry) as point_cnt, sum(a.passenger_count) as passenger_cnt, sum(a.trip_dist) as total_trip_dist, sum(a.fare) as total_fare FROM pickupInfoPartDf a group by a._id order by a._id asc")

		if i == 0:
			pickupInfoDf = pickupInfoPartDf
		else:
			pickupInfoDf.union(pickupInfoPartDf) 
	pickupInfoDf.createOrReplaceTempView("pickupInfoDf")
	pickupInfoDf.show(5, False)

	pickupInfoDf = spark.sql("SELECT a._id, sum(a.point_cnt) as point_cnt, sum(a.passenger_cnt) as passenger_cnt, sum(a.total_trip_dist) as total_trip_dist, sum(a.total_fare) as total_fare FROM pickupInfoDf a group by a._id order by a._id asc")
	pickupInfoDf.createOrReplaceTempView("pickupInfoDf")
	pickupInfoDf.show(5, False)

	#pickupInfoDf.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/processed_data/nyc_multivariate_data/nyc_trip_info_multi_df")
	print("Completed generating dataframe consisting of pickup information of selected attributes")


	# Generating multivariate grid dataset
	num_attrs = 4
	tripInfo = np.zeros(shape = (num_rows, num_cols, num_attrs))
	pickupInfo = pickupInfoDf.collect()
	ids = []
	for k in range(len(pickupInfo)):
		_id = pickupInfo[k][0]
		tripInfo[int(_id/num_cols)][_id%num_cols][0] = int(pickupInfo[k][1])
		tripInfo[int(_id/num_cols)][_id%num_cols][1] = int(pickupInfo[k][2])
		tripInfo[int(_id/num_cols)][_id%num_cols][2] = pickupInfo[k][3]
		tripInfo[int(_id/num_cols)][_id%num_cols][3] = pickupInfo[k][4]
		ids.append(_id)

	full_ids = ids.copy()
	priority_length = int(len(ids)/5)
	for _id in range(num_rows*num_cols):
		if _id not in ids:
			i = int(_id/num_cols)
			j = _id%num_cols

			neighbors = []
			if (i - 1) >= 0 and (i - 1) * num_cols + j in full_ids:
				neighbors.append((i - 1) * num_cols + j)
			if (i + 1) < num_rows and (i + 1) * num_cols + j in full_ids:
				neighbors.append((i + 1) * num_cols + j)
			if (j - 1) >= 0 and i * num_cols + (j - 1) in full_ids:
				neighbors.append(i * num_cols + (j - 1))
			if (j + 1) < num_cols and i * num_cols + (j + 1) in full_ids:
				neighbors.append(i * num_cols + (j + 1))

			new_ids = ids.copy()
			for m in range(priority_length):
				for k in range(len(neighbors)):
					new_ids.append(neighbors[k])

			random.shuffle(new_ids)
			selected_id = random.choice(new_ids)
			if selected_id != -1:
				tripInfo[int(_id/num_cols)][_id%num_cols] = tripInfo[int(selected_id/num_cols)][selected_id%num_cols]
				full_ids.append(_id)

	file_tripInfo = open("data/processed_data/taxi_trip_multivariate_data/taxi_trip_multivariate_grid.npy", "wb")
	np.save(file_tripInfo, tripInfo)
	print("Completed generating multivariate grid dataset")


	# Calculating centroids of grid cells
	poly_centroids = list(map(lambda x: x.centroid, grid_geom_list))
	centroids_list = list(map(lambda center_point: [center_point.x, center_point.y], poly_centroids))

	centroid_data =  []
	for k in range(len(centroids_list)):
		centroid_data.append([centroids_list[k][0], centroids_list[k][1], tripInfo[int(k/num_cols)][k % num_cols][0]])
	centroid_data = np.array(centroid_data)

	file_centroid_data = open("data/processed_data/taxi_trip_multivariate_data/centroid_data.npy", "wb")
	np.save(file_centroid_data, centroid_data)
	print("Completed calculating centroids of grid cells")



def process_taxi_trip_univariate_data(lat_interval, lon_interval):
	# Loading shape file containing spatial information
	gdf = load_dataset("data/taxi_trip/taxi_zones/taxi_zones.shp", True)
	print("Shape file loaded")

	# Creating grid partitions
	num_rows, num_cols, grid_geom_list = partition_by_grid(gdf, lat_interval, lon_interval)

	polyListDf = pd.DataFrame(grid_geom_list, columns=['geom'])
	polyListDf.to_csv("data/processed_data/taxi_trip_univariate_data/polygon_cells.csv")
	print("Grid partitions are created")

	# Defining schema of polygons
	schema = StructType(
		[
		StructField("_id", IntegerType(), False),
		StructField("geom", GeometryType(), False)
		]
		)

	# Creating list of polygons
	poly_data = []
	for i in range(len(grid_geom_list)):
		poly_data.append([i, grid_geom_list[i]])

	# Creating dataframes of polygons
	polyDf = spark.createDataFrame(poly_data, schema)
	polyDf.show(5, False)
	print("Polygon dataframe created")

	# Converting polygon dataframes into polygon RDDs
	polyRDD = Adapter.toSpatialRdd(polyDf, "geom")
	polyRDD.CRSTransform("epsg:2263", "epsg:2263")
	polyRDD.analyze()
	polyRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("Polygon dataframe converted into polygon RDD")


	# Loading CSV dataset
	tripDf = spark.read.format("csv").option("delimiter",",").option("header","true").load("data/taxi_trip/yellow_tripdata_2009-01.csv")
	tripDf = tripDf.withColumn("Serial_ID", monotonically_increasing_id())
	tripDf.createOrReplaceTempView("tripDf")
	tripDf.show(5, False)
	print("CSV dataset loaded")


	# Generating dataframe consisting of pickup information of selected attribute
	buildOnSpatialPartitionedRDD = True
	usingIndex = True
	considerBoundaryIntersection = True

	pickupInfoDf = spark.createDataFrame([], StructType([]))
	for i in range(15):
		start_id = i * 1000000
		end_id = (i + 1) * 1000000 - 1
		pointDf = spark.sql("select ST_Point(double(tripDf.Start_Lat), double(tripDf.Start_Lon)) as point_loc, int(tripDf.Passenger_Count) as passenger_count, float(tripDf.Trip_Distance) as trip_dist, float(tripDf.Fare_Amt) as fare from tripDf where tripDf.Serial_ID >= {0} and tripDf.Serial_ID <= {1}".format(start_id, end_id))
		pointRDD = Adapter.toSpatialRdd(pointDf, "point_loc")
		pointRDD.CRSTransform("epsg:4326", "epsg:2263")

		pointRDD.analyze()
		pointRDD.spatialPartitioning(polyRDD.getPartitioner())
		pointRDD.buildIndex(IndexType.QUADTREE, buildOnSpatialPartitionedRDD)
		result_pair_rdd = JoinQueryRaw.SpatialJoinQueryFlat(pointRDD, polyRDD, usingIndex, considerBoundaryIntersection)
		pickupInfoPartDf = Adapter.toDf(result_pair_rdd, polyRDD.fieldNames, pointRDD.fieldNames, spark)
		pickupInfoPartDf.createOrReplaceTempView("pickupInfoPartDf")

		pickupInfoPartDf = spark.sql("SELECT int(a._id) as _id, count(a.rightgeometry) as point_cnt, sum(a.passenger_count) as passenger_cnt, sum(a.trip_dist) as total_trip_dist, sum(a.fare) as total_fare FROM pickupInfoPartDf a group by a._id order by a._id asc")

		if i == 0:
			pickupInfoDf = pickupInfoPartDf
		else:
			pickupInfoDf.union(pickupInfoPartDf) 
	pickupInfoDf.createOrReplaceTempView("pickupInfoDf")
	pickupInfoDf.show(5, False)

	pickupInfoDf = spark.sql("SELECT a._id, sum(a.point_cnt) as point_cnt, sum(a.passenger_cnt) as passenger_cnt, sum(a.total_trip_dist) as total_trip_dist, sum(a.total_fare) as total_fare FROM pickupInfoDf a group by a._id order by a._id asc")
	pickupInfoDf.createOrReplaceTempView("pickupInfoDf")
	pickupInfoDf.show(5, False)

	#pickupInfoDf.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/processed_data/nyc_univariate_data/nyc_trip_info_uni_df")
	print("Completed generating dataframe consisting of pickup information of selected attributes")


	# Generating univariate grid dataset
	num_attrs = 1
	pickup_counts = np.zeros(shape = (num_rows, num_cols, num_attrs))
	ids = []
	pickupInfo = pickupInfoDf.collect()
	for k in range(len(pickupInfo)):
		_id = pickupInfo[k][0]
		pickup_counts[int(_id/num_cols)][_id%num_cols][0] = int(pickupInfo[k][1])
		ids.append(_id)

	full_ids = ids.copy()
	priority_length = int(len(ids)/5)
	for _id in range(num_rows*num_cols):
		if _id not in ids:
			i = int(_id/num_cols)
			j = _id%num_cols

			neighbors = []
			if (i - 1) >= 0 and (i - 1) * num_cols + j in full_ids:
				neighbors.append((i - 1) * num_cols + j)
			if (i + 1) < num_rows and (i + 1) * num_cols + j in full_ids:
				neighbors.append((i + 1) * num_cols + j)
			if (j - 1) >= 0 and i * num_cols + (j - 1) in full_ids:
				neighbors.append(i * num_cols + (j - 1))
			if (j + 1) < num_cols and i * num_cols + (j + 1) in full_ids:
				neighbors.append(i * num_cols + (j + 1))

			new_ids = ids.copy()
			for m in range(priority_length):
				for k in range(len(neighbors)):
					new_ids.append(neighbors[k])

			random.shuffle(new_ids)
			selected_id = random.choice(new_ids)
			if selected_id != -1:
				pickup_counts[int(_id/num_cols)][_id%num_cols] = pickup_counts[int(selected_id/num_cols)][selected_id%num_cols]
				full_ids.append(_id)

	file_pickup_count = open("data/processed_data/taxi_trip_univariate_data/taxi_trip_univariate_grid.npy", "wb")
	np.save(file_pickup_count, pickup_counts)
	print("Completed generating univariate grid dataset")


	# Calculating centroids of grid cells
	poly_centroids = list(map(lambda x: x.centroid, grid_geom_list))
	centroids_list = list(map(lambda center_point: [center_point.x, center_point.y], poly_centroids))

	centroid_data =  []
	for k in range(len(centroids_list)):
		centroid_data.append([centroids_list[k][0], centroids_list[k][1], pickup_counts[int(k/num_cols)][k % num_cols][0]])
	centroid_data = np.array(centroid_data)

	file_centroid_data = open("data/processed_data/taxi_trip_univariate_data/centroid_data.npy", "wb")
	np.save(file_centroid_data, centroid_data)
	print("Completed calculating centroids of grid cells")



def process_home_sales_multivariate_data(lat_interval, lon_interval):
	# Loading shape file containing spatial information
	gdf = load_dataset("data/home_sales/home_zones/kc_house.shp", True)
	gdf = gdf.to_crs('epsg:2263')
	print("Shape file loaded")

	# Creating grid partitions
	num_rows, num_cols, grid_geom_list = partition_by_grid(gdf, lat_interval, lon_interval)

	polyListDf = pd.DataFrame(grid_geom_list, columns=['geom'])
	polyListDf.to_csv("data/processed_data/home_sales_multivariate_data/polygon_cells.csv")
	print("Grid partitions are created")

	# Defining schema of polygons
	schema = StructType(
		[
		StructField("_id", IntegerType(), False),
		StructField("geom", GeometryType(), False)
		]
		)

	# Creating list of polygons
	poly_data = []
	for i in range(len(grid_geom_list)):
		poly_data.append([i, grid_geom_list[i]])

	# Creating dataframes of polygons
	polyDf = spark.createDataFrame(poly_data, schema)
	polyDf.show(5, False)
	print("Polygon dataframe created")

	# Converting polygon dataframes into polygon RDDs
	polyRDD = Adapter.toSpatialRdd(polyDf, "geom")
	polyRDD.CRSTransform("epsg:2263", "epsg:2263")
	polyRDD.analyze()
	polyRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("Polygon dataframe converted into polygon RDD")


	# Loading CSV dataset
	houseDf = spark.read.format("csv").option("delimiter",",").option("header","true").load("data/home_sales/kc_house_data.csv")
	houseDf = houseDf.withColumn("Serial_ID", monotonically_increasing_id())
	houseDf.createOrReplaceTempView("houseDf")
	houseDf.show(5, False)
	print("CSV dataset loaded")


	# Generating dataframe consisting of house sales information of selected attributes
	pointDf = spark.sql("select ST_Point(double(houseDf.lat), double(houseDf.long)) as point_loc, houseDf.price, houseDf.bedrooms, houseDf.bathrooms, houseDf.sqft_living, houseDf.sqft_lot, houseDf.yr_built, houseDf.yr_renovated from houseDf")
	pointDf = pointDf.withColumn("yr_renovated", when(col("yr_renovated") == 0, col("yr_built")).otherwise(col("yr_renovated")))
	pointDf = pointDf.withColumn("build_age", 2015 - col("yr_built"))
	pointDf = pointDf.withColumn("renov_age", 2015 - col("yr_renovated"))
	pointDf = pointDf.drop("yr_built")
	pointDf = pointDf.drop("yr_renovated")

	pointRDD = Adapter.toSpatialRdd(pointDf, "point_loc")
	pointRDD.CRSTransform("epsg:4326", "epsg:2263")
	pointRDD.analyze()
	pointRDD.spatialPartitioning(polyRDD.getPartitioner())

	buildOnSpatialPartitionedRDD = True
	usingIndex = True
	considerBoundaryIntersection = True

	pointRDD.buildIndex(IndexType.QUADTREE, buildOnSpatialPartitionedRDD)
	result_pair_rdd = JoinQueryRaw.SpatialJoinQueryFlat(pointRDD, polyRDD, usingIndex, considerBoundaryIntersection)
	houseInfoDf = Adapter.toDf(result_pair_rdd, polyRDD.fieldNames, pointRDD.fieldNames, spark)
	houseInfoDf.createOrReplaceTempView("houseInfoDf")

	houseInfoDf = spark.sql("SELECT int(a._id) as _id, avg(a.price) as avg_price, avg(a.bedrooms) as avg_bedrooms, avg(a.bathrooms) as avg_bathrooms, avg(a.sqft_living) as avg_sqft_living, avg(a.sqft_lot) as avg_sqft_lot, avg(a.build_age) as avg_build_age, avg(a.renov_age) as avg_renov_age FROM houseInfoDf a group by a._id order by a._id asc")
	houseInfoDf.createOrReplaceTempView("houseInfoDf")
	houseInfoDf.show(5, False)

	#houseInfoDf.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/processed_data/wa_multivariate_data/wa_house_info_df")
	print("Completed generating dataframe consisting of house sales information of selected attributes")


	# Generating multivariate grid dataset
	num_attrs = 7
	houseInfo = np.zeros(shape = (num_rows, num_cols, num_attrs))
	houseInfoCollection = houseInfoDf.collect()
	ids = []
	for k in range(len(houseInfoCollection)):
		_id = houseInfoCollection[k][0]
		for r in range(num_attrs):
			houseInfo[int(_id/num_cols)][_id%num_cols][r] = houseInfoCollection[k][r+1]
		ids.append(_id)

	full_ids = ids.copy()
	priority_length = int(len(ids)/5)
	for _id in range(num_rows*num_cols):
		if _id not in ids:
			i = int(_id/num_cols)
			j = _id%num_cols

			neighbors = []
			if (i - 1) >= 0 and (i - 1) * num_cols + j in full_ids:
				neighbors.append((i - 1) * num_cols + j)
			if (i + 1) < num_rows and (i + 1) * num_cols + j in full_ids:
				neighbors.append((i + 1) * num_cols + j)
			if (j - 1) >= 0 and i * num_cols + (j - 1) in full_ids:
				neighbors.append(i * num_cols + (j - 1))
			if (j + 1) < num_cols and i * num_cols + (j + 1) in full_ids:
				neighbors.append(i * num_cols + (j + 1))

			new_ids = ids.copy()
			for m in range(priority_length):
				for k in range(len(neighbors)):
					new_ids.append(neighbors[k])

			random.shuffle(new_ids)
			selected_id = random.choice(new_ids)
			if selected_id != -1:
				houseInfo[int(_id/num_cols)][_id%num_cols] = houseInfo[int(selected_id/num_cols)][selected_id%num_cols]
				full_ids.append(_id)

	file_houseInfo = open("data/processed_data/home_sales_multivariate_data/home_sales_multivariate_grid.npy", "wb")
	np.save(file_houseInfo, houseInfo)
	print("Completed generating multivariate grid dataset")


	# Calculating centroids of grid cells
	poly_centroids = list(map(lambda x: x.centroid, grid_geom_list))
	centroids_list = list(map(lambda center_point: [center_point.x, center_point.y], poly_centroids))

	centroid_data =  []
	for k in range(len(centroids_list)):
		centroid_data.append([centroids_list[k][0], centroids_list[k][1], houseInfo[int(k/num_cols)][k % num_cols][0]])
	centroid_data = np.array(centroid_data)

	file_centroid_data = open("data/processed_data/home_sales_multivariate_data/centroid_data.npy", "wb")
	np.save(file_centroid_data, centroid_data)
	print("Completed calculating centroids of grid cells")



def process_vehicles_univariate_data(lat_interval, lon_interval):
	# Loading CSV dataset
	carsDf = spark.read.format("csv").option("delimiter",",").option("header","true").load("data/chicago_vehicles/abandoned_cars.csv")
	carsDf = carsDf.dropna(subset=['Latitude'])
	carsDf = carsDf.dropna(subset=['Longitude'])
	carsDf = carsDf.withColumn("Serial_ID", monotonically_increasing_id())
	carsDf.createOrReplaceTempView("carsDf")
	carsDf.show(5, False)
	print("CSV dataset loaded")

	# Processing maximum and minimum of latitudes and longitudes
	lat_min = carsDf.agg({"Latitude": "min"}).collect()[0][0]
	lat_max = carsDf.agg({"Latitude": "max"}).collect()[0][0]
	lon_min = carsDf.agg({"Longitude": "min"}).collect()[0][0]
	lon_max = carsDf.agg({"Longitude": "max"}).collect()[0][0]

	inProj = Proj('epsg:4326')
	outProj = Proj('epsg:2263')
	lat_min, lon_min = transform(inProj, outProj, lat_min, lon_min)
	lat_max, lon_max = transform(inProj, outProj, lat_max, lon_max)

	# Creating grid partitions
	num_rows, num_cols, grid_geom_list = partition_by_grid_nonshape_file(lat_min, lat_max, lon_min, lon_max, lat_interval, lon_interval)
	#num_rows, num_cols, grid_geom_list = partition_by_grid_nonshape_file(lat_min, lat_max, lon_min, lon_max, -298, 545)
	#num_rows, num_cols, grid_geom_list = partition_by_grid_nonshape_file(lat_min, lat_max, lon_min, lon_max, -264, 480)

	polyListDf = pd.DataFrame(grid_geom_list, columns=['geom'])
	polyListDf.to_csv("data/processed_data/vehicles_univariate_data/polygon_cells.csv")
	print("Grid partitions are created")

	# Defining schema of polygons
	schema = StructType(
		[
		StructField("_id", IntegerType(), False),
		StructField("geom", GeometryType(), False)
		]
		)

	# Creating list of polygons
	poly_data = []
	for i in range(len(grid_geom_list)):
		poly_data.append([i, grid_geom_list[i]])

	# Creating dataframes of polygons
	polyDf = spark.createDataFrame(poly_data, schema)
	polyDf.show(5, False)
	print("Polygon dataframe created")

	# Converting polygon dataframes into polygon RDDs
	polyRDD = Adapter.toSpatialRdd(polyDf, "geom")
	polyRDD.CRSTransform("epsg:2263", "epsg:2263")
	polyRDD.analyze()
	polyRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("Polygon dataframe converted into polygon RDD")


	# Generating point dataframe
	pointDf = spark.sql("select ST_Point(double(carsDf.Latitude), double(carsDf.Longitude)) as point_loc from carsDf")

	# Converting point dataframe into point RDD
	pointRDD = Adapter.toSpatialRdd(pointDf, "point_loc")
	pointRDD.CRSTransform("epsg:4326", "epsg:2263")
	pointRDD.analyze()
	pointRDD.spatialPartitioning(polyRDD.getPartitioner())


	# Generating dataframe consisting of cars information of selected attribute
	buildOnSpatialPartitionedRDD = True
	usingIndex = True
	considerBoundaryIntersection = True

	pointRDD.buildIndex(IndexType.QUADTREE, buildOnSpatialPartitionedRDD)
	result_pair_rdd = JoinQueryRaw.SpatialJoinQueryFlat(pointRDD, polyRDD, usingIndex, considerBoundaryIntersection)
	carsInfoDf = Adapter.toDf(result_pair_rdd, polyRDD.fieldNames, pointRDD.fieldNames, spark)
	carsInfoDf.createOrReplaceTempView("carsInfoDf")

	carsInfoDf = spark.sql("SELECT int(a._id) as _id, count(a.rightgeometry) as cars_count FROM carsInfoDf a group by a._id order by a._id asc")
	carsInfoDf.createOrReplaceTempView("carsInfoDf")
	carsInfoDf.show(5, False)

	#carsInfoDf.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/processed_data/chicago_univariate_data/chicago_cars_info_df")
	print("Completed generating dataframe consisting of cars information of selected attributes")


	# Generating univariate grid dataset
	num_attrs = 1
	cars_counts = np.zeros(shape = (num_rows, num_cols, num_attrs))
	ids = []
	carsInfo = carsInfoDf.collect()
	for k in range(len(pickupInfo)):
		_id = carsInfo[k][0]
		cars_counts[int(_id/num_cols)][_id%num_cols][0] = int(carsInfo[k][1])
		ids.append(_id)

	full_ids = ids.copy()
	priority_length = int(len(ids)/5)
	for _id in range(num_rows*num_cols):
		if _id not in ids:
			i = int(_id/num_cols)
			j = _id%num_cols

			neighbors = []
			if (i - 1) >= 0 and (i - 1) * num_cols + j in full_ids:
				neighbors.append((i - 1) * num_cols + j)
			if (i + 1) < num_rows and (i + 1) * num_cols + j in full_ids:
				neighbors.append((i + 1) * num_cols + j)
			if (j - 1) >= 0 and i * num_cols + (j - 1) in full_ids:
				neighbors.append(i * num_cols + (j - 1))
			if (j + 1) < num_cols and i * num_cols + (j + 1) in full_ids:
				neighbors.append(i * num_cols + (j + 1))

			new_ids = ids.copy()
			for m in range(priority_length):
				for k in range(len(neighbors)):
					new_ids.append(neighbors[k])

			random.shuffle(new_ids)
			selected_id = random.choice(new_ids)
			if selected_id != -1:
				cars_counts[int(_id/num_cols)][_id%num_cols] = cars_counts[int(selected_id/num_cols)][selected_id%num_cols]
				full_ids.append(_id)

	file_cars_count = open("data/processed_data/vehicles_univariate_data/vehicles_univariate_grid.npy", "wb")
	np.save(file_cars_count, cars_counts)
	print("Completed generating univariate grid dataset")


	# Calculating centroids of grid cells
	poly_centroids = list(map(lambda x: x.centroid, grid_geom_list))
	centroids_list = list(map(lambda center_point: [center_point.x, center_point.y], poly_centroids))

	centroid_data =  []
	for k in range(len(centroids_list)):
		centroid_data.append([centroids_list[k][0], centroids_list[k][1], cars_counts[int(k/num_cols)][k % num_cols][0]])
	centroid_data = np.array(centroid_data)

	file_centroid_data = open("data/processed_data/vehicles_univariate_data/centroid_data.npy", "wb")
	np.save(file_centroid_data, centroid_data)
	print("Completed calculating centroids of grid cells")



def process_earning_multivariate_data(lat_interval, lon_interval):
	# Loading shape file containing spatial information
	gdf = load_dataset("data/nyc_earning/shape_file/NYC Area2010_2data.shp", True)
	print("Shape file loaded")

	# Creating grid partitions
	num_rows, num_cols, grid_geom_list = partition_by_grid(gdf, lat_interval, lon_interval)

	polyListDf = pd.DataFrame(grid_geom_list, columns=['geom'])
	polyListDf.to_csv("data/processed_data/earning_multivariate_data/polygon_cells.csv")
	print("Grid partitions are created")

	# Defining schema of polygons
	schema = StructType(
		[
		StructField("_id", IntegerType(), False),
		StructField("geom", GeometryType(), False)
		]
		)

	# Creating list of polygons
	poly_data = []
	for i in range(len(grid_geom_list)):
		poly_data.append([i, grid_geom_list[i]])

	# Creating dataframes of polygons
	polyDf = spark.createDataFrame(poly_data, schema)
	polyDf.show(5, False)
	print("Polygon dataframe created")

	# Converting polygon dataframes into polygon RDDs
	polyRDD = Adapter.toSpatialRdd(polyDf, "geom")
	polyRDD.analyze()
	polyRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("Polygon dataframe converted into polygon RDD")


	# Preaparing earning attributes for multivariate data
	gdf['CE01'] = gdf['CE01_02'] + gdf['CE01_03'] + gdf['CE01_04'] + gdf['CE01_05'] + gdf['CE01_06'] + gdf['CE01_07'] + gdf['CE01_08'] + gdf['CE01_09'] + gdf['CE01_10'] + gdf['CE01_11'] + gdf['CE01_12'] + gdf['CE01_13'] + gdf['CE01_14']
	gdf['CE02'] = gdf['CE02_02'] + gdf['CE02_03'] + gdf['CE02_04'] + gdf['CE02_05'] + gdf['CE02_06'] + gdf['CE02_07'] + gdf['CE02_08'] + gdf['CE02_09'] + gdf['CE02_10'] + gdf['CE02_11'] + gdf['CE02_12'] + gdf['CE02_13'] + gdf['CE02_14']
	gdf['CE03'] = gdf['CE03_02'] + gdf['CE03_03'] + gdf['CE03_04'] + gdf['CE03_05'] + gdf['CE03_06'] + gdf['CE03_07'] + gdf['CE03_08'] + gdf['CE03_09'] + gdf['CE03_10'] + gdf['CE03_11'] + gdf['CE03_12'] + gdf['CE03_13'] + gdf['CE03_14']

	land_area = gdf['ALAND10'].tolist()
	water_area = gdf['AWATER10'].tolist()
	job_1250 = gdf['CE01'].tolist()
	job_3333 = gdf['CE02'].tolist()
	job_3333_up = gdf['CE03'].tolist()
	blocks_geom = gdf['geometry'].tolist()

	schema_earning = StructType(
		[
		StructField("serial_id", IntegerType(), False),
        StructField("land_area", IntegerType(), False),
        StructField("water_area", IntegerType(), False),
        StructField("job_1250", IntegerType(), False),
        StructField("job_3333", IntegerType(), False),
        StructField("job_3333_up", IntegerType(), False),
        StructField("geometry", GeometryType(), False)
        ]
    )

    earning_data = []
    for i in range(len(blocks_geom)):
    	earning_data.append([i, land_area[i], water_area[i], job_1250[i], job_3333[i], job_3333_up[i], blocks_geom[i]])
    print("Prepared earning attributes for multivariate data")

    earningDf = spark.createDataFrame(earning_data, schema_earning)
    earningDf.createOrReplaceTempView("earningDf")
    earningDf.show(5, False)
	print("Earning multivariate dataframe created")

	# converting earning dataframe into RDD
	earningRDD = Adapter.toSpatialRdd(earningDf, "geometry")
	earningRDD.analyze()
	earningRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("earning dataframe has been converted to RDD")


	# Generating dataframe consisting of earning information of selected attributes
	buildOnSpatialPartitionedRDD = True
	usingIndex = True
	considerBoundaryIntersection = True

	earningRDD.analyze()
	earningRDD.spatialPartitioning(polyRDD.getPartitioner())
	earningRDD.buildIndex(IndexType.QUADTREE, buildOnSpatialPartitionedRDD)
	result_pair_rdd = JoinQueryRaw.SpatialJoinQueryFlat(earningRDD, polyRDD, usingIndex, considerBoundaryIntersection)
	earningInfoDf = Adapter.toDf(result_pair_rdd, polyRDD.fieldNames, earningRDD.fieldNames, spark)
	earningInfoDf.createOrReplaceTempView("earningInfoDf")

	earningInfoDf = spark.sql("SELECT int(a._id) as _id, avg(a.land_area) as land_area, avg(a.water_area) as water_area, avg(a.job_1250) as job_1250, avg(a.job_3333) as job_3333, avg(a.job_3333_up) as job_3333_up FROM earningInfoDf a group by a._id order by a._id asc")

	earningInfoDf.createOrReplaceTempView("earningInfoDf")
	earningInfoDf.show(5, False)

	#earningInfoDf.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/processed_data/earning_multivariate_data/nyc_earning_info_df")
	print("Completed generating dataframe consisting of earning information of selected attributes")


	# Generating multivariate grid dataset
	num_attrs = 5
	earning_multi = np.zeros(shape = (num_rows, num_cols, num_attrs))
	earningInfo = earningInfoDf.collect()
	ids = []
	for k in range(len(earningInfo)):
		_id = earningInfo[k][0]
		earning_multi[int(_id/num_cols)][_id%num_cols][0] = float(earningInfo[k][1])
		earning_multi[int(_id/num_cols)][_id%num_cols][1] = float(earningInfo[k][2])
		earning_multi[int(_id/num_cols)][_id%num_cols][2] = float(earningInfo[k][3])
		earning_multi[int(_id/num_cols)][_id%num_cols][3] = float(earningInfo[k][4])
		earning_multi[int(_id/num_cols)][_id%num_cols][4] = float(earningInfo[k][5])
		ids.append(_id)

	full_ids = ids.copy()
	priority_length = int(len(ids)/5)
	for _id in range(num_rows*num_cols):
		if _id not in ids:
			choice = random.randint(0, 20)
			if choice == 0:
				earning_multi[int(_id/num_cols)][_id%num_cols] = [0, 0, 0, 0, 0]
				continue

			i = int(_id/num_cols)
			j = _id%num_cols

			neighbors = []
			if (i - 1) >= 0 and (i - 1) * num_cols + j in full_ids:
				neighbors.append((i - 1) * num_cols + j)
			if (i + 1) < num_rows and (i + 1) * num_cols + j in full_ids:
				neighbors.append((i + 1) * num_cols + j)
			if (j - 1) >= 0 and i * num_cols + (j - 1) in full_ids:
				neighbors.append(i * num_cols + (j - 1))
			if (j + 1) < num_cols and i * num_cols + (j + 1) in full_ids:
				neighbors.append(i * num_cols + (j + 1))

			new_ids = ids.copy()
			for m in range(priority_length):
				for k in range(len(neighbors)):
					new_ids.append(neighbors[k])

			random.shuffle(new_ids)
			selected_id = random.choice(new_ids)
			if selected_id != -1:
				earning_multi[int(_id/num_cols)][_id%num_cols] = earning_multi[int(selected_id/num_cols)][selected_id%num_cols]
				full_ids.append(_id)

	file_earning_multi = open("data/processed_data/earning_multivariate_data/earning_multivariate_grid.npy", "wb")
	np.save(file_earning_multi, earning_multi)
	print("Completed generating multivariate grid dataset")


	# Calculating centroids of grid cells
	poly_centroids = list(map(lambda x: x.centroid, grid_geom_list))
	centroids_list = list(map(lambda center_point: [center_point.x, center_point.y], poly_centroids))

	centroid_data =  []
	for k in range(len(centroids_list)):
		centroid_data.append([centroids_list[k][0], centroids_list[k][1], earning_multi[int(k/num_cols)][k % num_cols][0]])
	centroid_data = np.array(centroid_data)

	file_centroid_data = open("data/processed_data/earning_multivariate_data/centroid_data.npy", "wb")
	np.save(file_centroid_data, centroid_data)
	print("Completed calculating centroids of grid cells")



def process_earning_univariate_data(lat_interval, lon_interval):
	# Loading shape file containing spatial information
	gdf = load_dataset("data/nyc_earning/shape_file/NYC Area2010_2data.shp", True)
	print("Shape file loaded")

	# Creating grid partitions
	num_rows, num_cols, grid_geom_list = partition_by_grid(gdf, lat_interval, lon_interval)

	polyListDf = pd.DataFrame(grid_geom_list, columns=['geom'])
	polyListDf.to_csv("data/processed_data/earning_univariate_data/polygon_cells.csv")
	print("Grid partitions are created")

	# Defining schema of polygons
	schema = StructType(
		[
		StructField("_id", IntegerType(), False),
		StructField("geom", GeometryType(), False)
		]
		)

	# Creating list of polygons
	poly_data = []
	for i in range(len(grid_geom_list)):
		poly_data.append([i, grid_geom_list[i]])

	# Creating dataframes of polygons
	polyDf = spark.createDataFrame(poly_data, schema)
	polyDf.show(5, False)
	print("Polygon dataframe created")

	# Converting polygon dataframes into polygon RDDs
	polyRDD = Adapter.toSpatialRdd(polyDf, "geom")
	polyRDD.analyze()
	polyRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("Polygon dataframe converted into polygon RDD")


	# Preaparing earning attributes
	gdf['CE01'] = gdf['CE01_02'] + gdf['CE01_03'] + gdf['CE01_04'] + gdf['CE01_05'] + gdf['CE01_06'] + gdf['CE01_07'] + gdf['CE01_08'] + gdf['CE01_09'] + gdf['CE01_10'] + gdf['CE01_11'] + gdf['CE01_12'] + gdf['CE01_13'] + gdf['CE01_14']
	gdf['CE02'] = gdf['CE02_02'] + gdf['CE02_03'] + gdf['CE02_04'] + gdf['CE02_05'] + gdf['CE02_06'] + gdf['CE02_07'] + gdf['CE02_08'] + gdf['CE02_09'] + gdf['CE02_10'] + gdf['CE02_11'] + gdf['CE02_12'] + gdf['CE02_13'] + gdf['CE02_14']
	gdf['CE03'] = gdf['CE03_02'] + gdf['CE03_03'] + gdf['CE03_04'] + gdf['CE03_05'] + gdf['CE03_06'] + gdf['CE03_07'] + gdf['CE03_08'] + gdf['CE03_09'] + gdf['CE03_10'] + gdf['CE03_11'] + gdf['CE03_12'] + gdf['CE03_13'] + gdf['CE03_14']

	land_area = gdf['ALAND10'].tolist()
	water_area = gdf['AWATER10'].tolist()
	job_1250 = gdf['CE01'].tolist()
	job_3333 = gdf['CE02'].tolist()
	job_3333_up = gdf['CE03'].tolist()
	blocks_geom = gdf['geometry'].tolist()

	schema_earning = StructType(
		[
		StructField("serial_id", IntegerType(), False),
        StructField("land_area", IntegerType(), False),
        StructField("water_area", IntegerType(), False),
        StructField("job_1250", IntegerType(), False),
        StructField("job_3333", IntegerType(), False),
        StructField("job_3333_up", IntegerType(), False),
        StructField("geometry", GeometryType(), False)
        ]
    )

    earning_data = []
    for i in range(len(blocks_geom)):
    	earning_data.append([i, land_area[i], water_area[i], job_1250[i], job_3333[i], job_3333_up[i], blocks_geom[i]])
    print("Prepared earning attributes")

    earningDf = spark.createDataFrame(earning_data, schema_earning)
    earningDf.createOrReplaceTempView("earningDf")
    earningDf.show(5, False)
	print("Earning dataframe created")

	# converting earning dataframe into RDD
	earningRDD = Adapter.toSpatialRdd(earningDf, "geometry")
	earningRDD.analyze()
	earningRDD.spatialPartitioning(GridType.KDBTREE, 4)
	print("earning dataframe has been converted to RDD")


	# Generating dataframe consisting of earning information of selected attributes
	buildOnSpatialPartitionedRDD = True
	usingIndex = True
	considerBoundaryIntersection = True

	earningRDD.analyze()
	earningRDD.spatialPartitioning(polyRDD.getPartitioner())
	earningRDD.buildIndex(IndexType.QUADTREE, buildOnSpatialPartitionedRDD)
	result_pair_rdd = JoinQueryRaw.SpatialJoinQueryFlat(earningRDD, polyRDD, usingIndex, considerBoundaryIntersection)
	earningInfoDf = Adapter.toDf(result_pair_rdd, polyRDD.fieldNames, earningRDD.fieldNames, spark)
	earningInfoDf.createOrReplaceTempView("earningInfoDf")

	earningInfoDf = spark.sql("SELECT int(a._id) as _id, avg(a.land_area) as land_area, avg(a.water_area) as water_area, avg(a.job_1250) as job_1250, avg(a.job_3333) as job_3333, avg(a.job_3333_up) as job_3333_up FROM earningInfoDf a group by a._id order by a._id asc")

	earningInfoDf.createOrReplaceTempView("earningInfoDf")
	earningInfoDf.show(5, False)

	#earningInfoDf.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/processed_data/earning_multivariate_data/nyc_earning_info_df")
	print("Completed generating dataframe consisting of earning information of selected attribute")


	# Generating multivariate grid dataset
	num_attrs = 1
	earning_single = np.zeros(shape = (num_rows, num_cols, num_attrs))
	earningInfo = earningInfoDf.collect()
	ids = []
	for k in range(len(earningInfo)):
		_id = earningInfo[k][0]
		earning_single[int(_id/num_cols)][_id%num_cols][0] = float(earningInfo[k][3]) + float(earningInfo[k][4]) + float(earningInfo[k][5])
		ids.append(_id)

	full_ids = ids.copy()
	priority_length = int(len(ids)/5)
	for _id in range(num_rows*num_cols):
		if _id not in ids:
			choice = random.randint(0, 20)
			if choice == 0:
				earning_single[int(_id/num_cols)][_id%num_cols] = [0, 0, 0, 0, 0]
				continue

			i = int(_id/num_cols)
			j = _id%num_cols

			neighbors = []
			if (i - 1) >= 0 and (i - 1) * num_cols + j in full_ids:
				neighbors.append((i - 1) * num_cols + j)
			if (i + 1) < num_rows and (i + 1) * num_cols + j in full_ids:
				neighbors.append((i + 1) * num_cols + j)
			if (j - 1) >= 0 and i * num_cols + (j - 1) in full_ids:
				neighbors.append(i * num_cols + (j - 1))
			if (j + 1) < num_cols and i * num_cols + (j + 1) in full_ids:
				neighbors.append(i * num_cols + (j + 1))

			new_ids = ids.copy()
			for m in range(priority_length):
				for k in range(len(neighbors)):
					new_ids.append(neighbors[k])

			random.shuffle(new_ids)
			selected_id = random.choice(new_ids)
			if selected_id != -1:
				earning_single[int(_id/num_cols)][_id%num_cols] = earning_single[int(selected_id/num_cols)][selected_id%num_cols]
				full_ids.append(_id)

	file_earning_single = open("data/processed_data/earning_univariate_data/earning_univariate_grid.npy", "wb")
	np.save(file_earning_single, earning_single)
	print("Completed generating multivariate grid dataset")


	# Calculating centroids of grid cells
	poly_centroids = list(map(lambda x: x.centroid, grid_geom_list))
	centroids_list = list(map(lambda center_point: [center_point.x, center_point.y], poly_centroids))

	centroid_data =  []
	for k in range(len(centroids_list)):
		centroid_data.append([centroids_list[k][0], centroids_list[k][1], earning_single[int(k/num_cols)][k % num_cols][0]])
	centroid_data = np.array(centroid_data)

	file_centroid_data = open("data/processed_data/earning_univariate_data/centroid_data.npy", "wb")
	np.save(file_centroid_data, centroid_data)
	print("Completed calculating centroids of grid cells")