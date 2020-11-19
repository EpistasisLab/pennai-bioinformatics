import sklearn
import numpy as np
est = sklearn.linear_model.LogisticRegression()
hyper_params = [{
 'C': np.logspace(-4,4,10) ,
 'penalty': ['l1', 'l2'] ,
 'fit_intercept': [True, False] ,
 'dual': [False] ,
},{
 'C': np.logspace(-4,4,10) ,
 'penalty': ['l2'] ,
 'fit_intercept': [True, False] ,
 'dual': [True] ,
}]
