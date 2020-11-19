import sklearn
est = sklearn.ensemble.AdaBoostClassifier()
hyper_params = {
                'learning_rate': [0.1, 1.0, 0.5, 100.0, 10.0, 0.01, 50.0] ,
                'n_estimators': [100, 1000, 10, 50, 500] ,
                }
