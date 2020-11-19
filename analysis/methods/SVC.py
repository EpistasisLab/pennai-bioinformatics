import sklearn
est = sklearn.svm.SVC()
hyper_params = [ {
 'C': [0.01,0.1,1],
 'gamma': ['auto','scale'] ,
 'kernel': ['poly'] ,
 'degree': [2, 3] ,
 'coef0': [0.0, 0.5, 1.0] ,
 'max_iter':[5000]
},
{
 'C': [0.01,0.1,1],
 'gamma': ['auto','scale'] ,
 'kernel': ['rbf'] ,
 'max_iter':[5000]
} ]
