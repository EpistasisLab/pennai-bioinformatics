import sklearn.svm

est = sklearn.svm.SVR(cache_size=700, max_iter=10000)
hyper_params = {
	 'kernel': ['poly', 'rbf'],
	'tol': [1e-05, 0.0001, 0.001, 0.01, 0.1],
	'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
	'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
	'degree': [2, 3],
	'coef0': [0.0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
	}

