import sklearn.kernel_ridge
est = sklearn.kernel_ridge.KernelRidge(kernel='rbf')
hyper_params = {
	 'alpha': [0.001, 0.01, 0.1, 1],
	'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
	}
