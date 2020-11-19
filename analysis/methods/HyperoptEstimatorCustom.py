# William La Cava
# setup for hyperopt sklearn estimator. 
from hyperopt import hp
from hpsklearn import HyperoptEstimator, ada_boost, decision_tree, extra_trees, \
        gaussian_nb, gradient_boosting, knn, random_forest, sgd
from hyperopt import tpe
import numpy as np

## configure different estimators to match the options in PennAI
################################################################
# Adaboost
learning_rate = hp.choice('adaboost_learning_rate', [0.1, 1.0, 0.5, 100.0, 10.0, 0.01, 50.0]) 
n_estimators = hp.choice('adaboost_n_estimators', [100, 1000, 10, 50, 500]) 
my_adaboost = ada_boost('my_adaboost', learning_rate=learning_rate, n_estimators=n_estimators)

# Decision Tree Classifier
# min_weight_fraction_leaf = hp.choice('dt_min_weight_fraction_leaf',
#     [0.0, 0.35, 0.45, 0.05, 0.15, 0.2, 0.5, 0.1, 0.4, 0.3, 0.25]) ,
max_features = hp.choice('dt_max_features',[0.75, 0.25, 0.5, 0.1, None, 'sqrt', 'log2'])
criterion = hp.choice('dt_criterion',['gini', 'entropy'])
my_dt = decision_tree('my_dt', max_features = max_features, criterion = criterion)

# Extra trees
n_estimators = hp.choice('et_n_estimators',[100, 1000, 10, 50, 500]) 
# min_weight_fraction_leaf = hp.choice('et_min_weight_fraction_leaf',
#                 [0.25, 0.4, 0.1, 0.0, 0.15, 0.5, 0.05, 0.45, 0.3, 0.2, 0.35]) ,
max_features = hp.choice('et_max_features',[0.1, 0.25, 0.5, 0.75, None, 'sqrt', 'log2'])
criterion = hp.choice('et_criterion',['gini', 'entropy'])
my_extratrees = extra_trees('my_extra_trees', n_estimators=n_estimators, 
        max_features = max_features, criterion=criterion)

# GaussianNB
# var_smoothing = hp.choice('nb_var_smoothing',np.logspace(-9,-3,7))
my_gaussiannb = gaussian_nb('my_gaussian_nb')

# gradient boosting classifier
loss = hp.choice('gbc_loss',['deviance']) 
learning_rate = hp.choice('gb_learning_rate',[0.1, 1.0, 0.5, 10.0, 0.01] )
n_estimators = hp.choice('gb_n_estimators', [100, 1000, 10, 50, 500] )
max_depth = hp.choice('gb_max_depth',[1, 2, 3, 4, 5, None, 10, 50, 20]) 
max_features = hp.choice('gb_max_features',[None, 'sqrt', 'log2']) 
my_gb = gradient_boosting('my_gb', loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
        max_depth=max_depth, max_features=max_features)

# kneighbors
n_neighbors = hp.choice('knn_n_neighbors',np.linspace(1,19,19,dtype=int))
weights = hp.choice('knn_weights',['uniform', 'distance']) 
my_knn = knn('my_knn',n_neighbors=n_neighbors,weights=weights)

#logistic regression
# not supported!
C = hp.choice('lr_dual_true_c',np.logspace(-4,4,10)) 
penalty = hp.choice('lr_dual_true_penalty',['l2']) 
fit_intercept = hp.choice('lr_dual_true_fit_intercept',[True, False]) 

# random forest
n_estimators = hp.choice('rf_n_estimators',[100, 1000, 10, 50, 500]) 
# min_weight_fraction_leaf = hp.choice('rf_min_weight_fraction_leaf',
#                 [0.25, 0.4, 0.1, 0.0, 0.15, 0.5, 0.05, 0.45, 0.3, 0.2, 0.35] ),
max_features = hp.choice('rf_max_features',[0.1, 0.25, 0.5, 0.75, None, 'sqrt', 'log2']) 
criterion = hp.choice('rf_criterion',['gini', 'entropy']) 
my_rf = random_forest('my_rf', n_estimators=n_estimators, 
        max_features = max_features, criterion=criterion)

# SGD
loss = hp.choice('sgd_loss',['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron']) 
penalty = hp.choice('sgd_penalty',['elasticnet']) 
alpha =  hp.choice('sgd_alpha',[0.0001, 0.001, 0.01]) 
learning_rate = hp.choice('sgd_learning_rate',['invscaling', 'constant']) 
fit_intercept = hp.choice('sgd_fit_intercept',[True, False]) 
l1_ratio = hp.choice('sgd_l1_ratio',[ 0.0, 0.25, 0.5, 0.75, 1.0]) 
eta0 = hp.choice('sgd_eta0',[ 0.01, 0.1, 1.0]) 
power_t = hp.choice('sgd_power_t',[0.0, 0.1, 0.5, 1.0, 10.0, 100.0]) 
mysgd = sgd('my_sgd', penalty=penalty, loss=loss, alpha=alpha, learning_rate=learning_rate,
        fit_intercept=fit_intercept, l1_ratio=l1_ratio, eta0 = eta0, power_t=power_t)

################################################################

# combine all classifiers into one search space
clf = hp.choice( 'my_clf', 
            [my_adaboost, my_dt, my_extratrees, my_gaussiannb, my_gb, my_knn, my_rf, mysgd]
)
total_configs = 10000

est = HyperoptEstimator(classifier=clf,
                            algo=tpe.suggest, 
                            max_evals=total_configs, 
                            #trial_timeout=600)
                            trial_timeout=43200)
hyper_params = {'max_evals': [1,10,100,1000,10000]}
#hyper_params = {'max_evals': [10,100]}

