
from data_utilities import *

def main():
	# Preprocess NYC taxi trip multivariate dataset
	process_taxi_trip_multivariate_data(800, 800) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess NYC taxi trip univariate dataset
	process_taxi_trip_univariate_data(800, 800) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess Washington King county home sales multivariate dataset
	process_home_sales_multivariate_data(1616, 1616) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess Chicago abandoned cars univariate dataset
	process_vehicles_univariate_data(-433, 800) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess NYC earning multivariate dataset
	process_earning_multivariate_data(1025, 700) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess NYC earning univariate dataset
	process_earning_univariate_data(1025, 700) # arguments represent intervals for latitude and longitude respectively in the grid


if __name__ == "__main__":
	main()
