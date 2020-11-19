import sklearn
import numpy as np
est = sklearn.neighbors.KNeighborsClassifier()
hyper_params = {
    'n_neighbors': np.linspace(1,19,19,dtype=int) ,
    'weights': ['uniform', 'distance'] ,
}
