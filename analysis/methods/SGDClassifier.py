from sklearn import linear_model
hyper_params = {
 'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'] ,
 'penalty': ['elasticnet'] ,
 'alpha': [0.0001, 0.001, 0.01] ,
 'learning_rate': ['invscaling', 'constant'] ,
 'fit_intercept': [True, False] ,
 'l1_ratio': [ 0.0, 0.25, 0.5, 0.75, 1.0] ,
 'eta0': [ 0.01, 0.1, 1.0] ,
 'power_t': [0.0, 0.1, 0.5, 1.0, 10.0, 100.0] ,
}
est=linear_model.SGDClassifier()
