from sklearn import ensemble

hyper_params = {
	 'n_estimators': [100, 500],
	'criterion': ['mse', 'mae'],
	'max_features': ['sqrt', 'log2'],
	'min_samples_split': [2, 5, 10, 20],
	'min_samples_leaf': [1, 5, 10, 20],
	'bootstrap': ['true', 'false'],
	'min_weight_fraction_leaf': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45] 
	}

est=ensemble.RandomForestRegressor()
