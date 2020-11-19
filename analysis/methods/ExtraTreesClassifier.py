import sklearn
est = sklearn.ensemble.ExtraTreesClassifier()
hyper_params = {
    'n_estimators': [100, 1000, 10, 50, 500] ,
    'min_weight_fraction_leaf': [0.25, 0.4, 0.1, 0.0, 0.15, 0.5, 
        0.05, 0.45, 0.3, 0.2, 0.35] ,
    'max_features': [0.1, 0.25, 0.5, 0.75, None, 'sqrt', 'log2'] ,
    'criterion': ['gini', 'entropy'] ,
}
