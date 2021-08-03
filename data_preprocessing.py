
from data_utilities import *

def main():
	# Preprocess NYC taxi trip multivariate dataset
	process_nyc_multivariate_data(5000, 5000) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess NYC taxi trip univariate dataset
	process_nyc_univariate_data(5000, 5000) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess Washington King county home sales multivariate dataset
	process_wa_multivariate_data(1616, 1616) # arguments represent intervals for latitude and longitude respectively in the grid

	# Preprocess Chicago abandoned cars univariate dataset
	process_chicago_univariate_data(-433, 800) # arguments represent intervals for latitude and longitude respectively in the grid


if __name__ == "__main__":
	main()