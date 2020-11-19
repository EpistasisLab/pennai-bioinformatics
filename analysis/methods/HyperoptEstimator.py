# William La Cava
# setup for hyperopt sklearn estimator. 
from hyperopt import hp
from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe
import numpy as np

total_configs = 10000

est = HyperoptEstimator(classifier=any_classifier('my_clf'),
                            algo=tpe.suggest, 
                            max_evals=total_configs, 
                            #trial_timeout=600)
                            trial_timeout=43200)
hyper_params = {'max_evals': [1,10,100,1000,10000]}

