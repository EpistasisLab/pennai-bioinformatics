# William La Cava
# setup for autosklearn estimator. 
from autosklearn.classification import AutoSklearnClassifier
import numpy as np

total_configs = 10000
est = AutoSklearnClassifier(n_jobs=1)
# hyper_params = {'time_left_for_this_task': [3600,21600,43200]}
hyper_params = {'time_left_for_this_task': [600, 3600, 36000]}
#hyper_params = {'time_left_for_this_task': [30,60,90]}
