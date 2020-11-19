import sklearn
est = sklearn.tree.DecisionTreeClassifier()
hyper_params = {
    'min_weight_fraction_leaf': [0.0, 0.35, 0.45, 0.05, 0.15, 0.2, 0.5, 0.1, 0.4, 0.3, 0.25] ,
    'max_features': [0.75, 0.25, 0.5, 0.1, None, 'sqrt', 'log2'] ,
    'criterion': ['gini', 'entropy'] ,
    }
