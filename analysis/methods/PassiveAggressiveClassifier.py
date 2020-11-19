import sklearn
est = sklearn.linear_model.PassiveAggressiveClassifier()
hyper_params = {
 'C': [0.0, 1.0, 0.5, 0.1, 100.0, 10.0, 0.01, 50.0, 0.001] ,
 'loss': ['squared_hinge', 'hinge'] ,
 'fit_intercept': [True,False] ,
}
