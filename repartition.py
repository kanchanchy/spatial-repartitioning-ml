
from repartition_utilities import *


def main():

	# Starting repartitioning of NYC taxi trip multivariate data with infoloss threshold 0.05
	nyc_mulivariate_data_types = [DATA_TYPE_INT, DATA_TYPE_INT, DATA_TYPE_FLOAT_DOUBLE, DATA_TYPE_FLOAT_DOUBLE]
	infoLossThreshold_NYC_Multi = 0.05
	outputPath_NYC_Multi = "data/repartitioned_data/nyc_multivariate_data"
	file_nyc_multivariate = open("data/processed/nyc_multivariate/nyc_multivariate_grid.npy", "rb")
	nyc_multivariate_grid = np.load(file_nyc_multivariate)
	doRepartitioningMultiAttr(nyc_multivariate_grid, nyc_mulivariate_data_types, infoLossThreshold_NYC_Multi, outputPath_NYC_Multi)


	# Starting repartitioning of NYC taxi trip univariate data with infoloss threshold 0.05
	nyc_univariate_data_type = [DATA_TYPE_INT]
	infoLossThreshold_NYC_Uni = 0.05
	outputPath_NYC_Uni = "data/repartitioned_data/nyc_univariate_data"
	file_nyc_univariate = open("data/processed/nyc_univariate/nyc_univariate_grid.npy", "rb")
	nyc_univariate_grid = np.load(file_nyc_univariate)
	doRepartitioningMultiAttr(nyc_univariate_grid, nyc_univariate_data_type, infoLossThreshold_NYC_Uni, outputPath_NYC_Uni)


	# Starting repartitioning of WA King county home sales multivariate data with infoloss threshold 0.05
	wa_mulivariate_data_types = [DATA_TYPE_FLOAT_DOUBLE]*7
	infoLossThreshold_WA_Multi = 0.05
	outputPath_WA_Multi = "data/repartitioned_data/wa_multivariate_data"
	file_wa_multivariate = open("data/processed/wa_multivariate/wa_multivariate_grid.npy", "rb")
	wa_multivariate_grid = np.load(file_wa_multivariate)
	doRepartitioningMultiAttr(wa_multivariate_grid, wa_mulivariate_data_types, infoLossThreshold_WA_Multi, outputPath_WA_Multi)


	# Starting repartitioning of NYC taxi trip univariate data with infoloss threshold 0.05
	chicago_univariate_data_type = [DATA_TYPE_INT]
	infoLossThreshold_Chicago_Uni = 0.05
	outputPath_Chicago_Uni = "data/repartitioned_data/chicago_univariate_data"
	file_chicago_univariate = open("data/processed/chicago_univariate/chicago_univariate_grid.npy", "rb")
	chicago_univariate_grid = np.load(file_chicago_univariate)
	doRepartitioningMultiAttr(chicago_univariate_grid, chicago_univariate_data_type, infoLossThreshold_Chicago_Uni, outputPath_Chicago_Uni)


if __name__ == "__main__":
	main()