import xgboost
est = xgboost.XGBRegressor(objective='reg:squarederror')
hyper_params = {
	 'n_estimators': [100, 500],
	'learning_rate': [0.01, 0.1, 1],
	'max_depth': [1, 3, 5, 10],
	}

