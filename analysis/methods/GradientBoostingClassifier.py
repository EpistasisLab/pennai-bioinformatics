from sklearn import ensemble
hyper_params = {
'loss': ['deviance'] ,
'learning_rate': [0.1, 1.0, 0.5, 10.0, 0.01] ,
'n_estimators': [100, 1000, 10, 50, 500] ,
'max_depth': [1, 2, 3, 4, 5, None, 10, 50, 20] ,
'max_features': [None, 'sqrt', 'log2'] ,
}
est=ensemble.GradientBoostingClassifier()
