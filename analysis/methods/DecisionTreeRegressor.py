import sklearn.tree
est = sklearn.tree.DecisionTreeRegressor()
hyper_params = {
	 'criterion': ['mse', 'mae'],
	'max_depth': [3, 5, 10],
	'min_samples_split': [2, 5, 10, 20],
	'min_samples_leaf': [1, 5, 10, 20],
	'min_weight_fraction_leaf': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
	'max_features': ['sqrt', 'log2', None]
	}


