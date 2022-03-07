
from repartition_utilities import *


def main():

	# Starting repartitioning of NYC taxi trip multivariate data with infoloss threshold 0.05
	taxi_trip_mulivariate_data_types = [DATA_TYPE_INT, DATA_TYPE_INT, DATA_TYPE_FLOAT_DOUBLE, DATA_TYPE_FLOAT_DOUBLE]
	infoLossThreshold_taxi_trip_multi = 0.05
	outputPath_taxi_trip_multi = "data/repartitioned_data/taxi_trip_multivariate_data"
	file_taxi_trip_multivariate = open("data/processed_data/taxi_trip_multivariate_data/taxi_trip_multivariate_grid.npy", "rb")
	taxi_trip_multivariate_grid = np.load(file_taxi_trip_multivariate)
	doRepartitioningmultiAttr(taxi_trip_multivariate_grid, taxi_trip_mulivariate_data_types, infoLossThreshold_taxi_trip_multi, outputPath_taxi_trip_multi)


	# Starting repartitioning of NYC taxi trip univariate data with infoloss threshold 0.05
	taxi_trip_univariate_data_type = [DATA_TYPE_INT]
	infoLossThreshold_taxi_trip_uni = 0.05
	outputPath_taxi_trip_uni = "data/repartitioned_data/taxi_trip_univariate_data"
	file_taxi_trip_univariate = open("data/processed_data/taxi_trip_univariate_data/taxi_trip_univariate_grid.npy", "rb")
	taxi_trip_univariate_grid = np.load(file_taxi_trip_univariate)
	doRepartitioningmultiAttr(taxi_trip_univariate_grid, taxi_trip_univariate_data_type, infoLossThreshold_taxi_trip_uni, outputPath_taxi_trip_uni)


	# Starting repartitioning of WA King county home sales multivariate data with infoloss threshold 0.05
	home_sales_mulivariate_data_types = [DATA_TYPE_FLOAT_DOUBLE]*7
	infoLossThreshold_home_sales_multi = 0.05
	outputPath_home_salesmulti = "data/repartitioned_data/home_sales_multivariate_data"
	file_home_sales_multivariate = open("data/processed_data/home_sales_multivariate_data/home_sales_multivariate_grid.npy", "rb")
	home_sales_multivariate_grid = np.load(file_home_sales_multivariate)
	doRepartitioningmultiAttr(home_sales_multivariate_grid, home_sales_mulivariate_data_types, infoLossThreshold_home_sales_multi, outputPath_home_sales_multi)


	# Starting repartitioning of NYC taxi trip univariate data with infoloss threshold 0.05
	vehicles_univariate_data_type = [DATA_TYPE_INT]
	infoLossThreshold_vehicles_uni = 0.05
	outputPath_vehicles_uni = "data/repartitioned_data/vehicles_univariate_data"
	file_vehicles_univariate = open("data/processed_data/vehicles_univariate_data/vehicles_univariate_grid.npy", "rb")
	vehicles_univariate_grid = np.load(file_vehicles_univariate)
	doRepartitioningmultiAttr(vehicles_univariate_grid, vehicles_univariate_data_type, infoLossThreshold_vehicles_uni, outputPath_vehicles_uni)

	# Starting repartitioning of NYC earning multivariate data with infoloss threshold 0.05
	earning_mulivariate_data_types = [DATA_TYPE_FLOAT_DOUBLE, DATA_TYPE_FLOAT_DOUBLE, DATA_TYPE_FLOAT_DOUBLE, DATA_TYPE_FLOAT_DOUBLE, DATA_TYPE_FLOAT_DOUBLE]
	infoLossThreshold_earning_multi = 0.05
	outputPath_earning_multi = "data/repartitioned_data/earning_multivariate_data"
	file_earning_multivariate = open("data/processed_data/earning_multivariate_data/earning_multivariate_grid.npy", "rb")
	earning_multivariate_grid = np.load(file_earning_multivariate)
	doRepartitioningmultiAttr(earning_multivariate_grid, earning_mulivariate_data_types, infoLossThreshold_earning_multi, outputPath_earning_multi)

	# Starting repartitioning of NYC earning univariate data with infoloss threshold 0.05
	earning_univariate_data_types = [DATA_TYPE_FLOAT_DOUBLE]
	infoLossThreshold_earning_uni = 0.05
	outputPath_earning_uni = "data/repartitioned_data/earning_univariate_data"
	file_earning_univariate = open("data/processed_data/earning_univariate_data/earning_univariate_grid.npy", "rb")
	earning_univariate_grid = np.load(file_earning_univariate)
	doRepartitioningmultiAttr(earning_univariate_grid, earning_univariate_data_types, infoLossThreshold_earning_uni, outputPath_earning_uni)


if __name__ == "__main__":
	main()