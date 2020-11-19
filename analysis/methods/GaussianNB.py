import sklearn
est = sklearn.naive_bayes.GaussianNB()
hyper_params = {
        'var_smoothing':np.log(-9,-3,7)
}
