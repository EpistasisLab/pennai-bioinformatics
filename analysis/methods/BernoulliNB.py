import sklearn
est = sklearn.naive_bayes.BernoulliNB()
hyper_params = {
'alpha': [0.0, 0.25, 1.0, 0.1, 0.5, 5.0, 0.75, 10.0, 50.0, 25.0] ,
'fit_prior': ['true', 'false'] ,
'binarize': [0.0, 1.0, 0.5, 0.75, 0.9, 0.25, 0.1] ,
}
