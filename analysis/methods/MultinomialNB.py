import sklearn
est = sklearn.naive_bayes.MultinomialNB()
hyper_params = {
 'alpha': [0.0, 0.5, 0.1, 0.75, 0.25, 5.0, 1.0, 10.0, 50.0, 25.0] ,
 'fit_prior': ['true', 'false'] ,
}
