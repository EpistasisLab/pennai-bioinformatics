# William La Cava
# setup for autosklearn estimator. 
from autosklearn.classification import AutoSklearnClassifier
# from autosklearn.pipeline.components.feature_preprocessing.no_preprocessing import NoPreprocessing

import numpy as np

est = AutoSklearnClassifier(n_jobs=1, 
        include_preprocessors = ['no_preprocessing'],
        ensemble_size=1)
# hyper_params = {'time_left_for_this_task': [3600,21600,43200]}
hyper_params = {'time_left_for_this_task': [60, 600, 3600, 36000]}
# hyper_params = {'time_left_for_this_task': [60]}
