import sklearn.neighbors
import numpy as np
est = sklearn.neighbors.KNeighborsRegressor()
hyper_params = {
	 'n_neighbors': [1, 3, 5, 7, 9, 11],
	'weights': ['uniform', 'distance'],
	'p': [1, 2]
	}

