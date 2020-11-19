import sklearn.linear_model
est = sklearn.linear_model.LassoLarsCV(max_iter=10000)
hyper_params = {
	 'fit_intercept': ['true', 'false'],
	'normalize': ['true', 'false']
	}
