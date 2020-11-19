import sys
import itertools
import pandas as pd
from sklearn.model_selection import (KFold, ParameterGrid, cross_validate)
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score, make_scorer
from sklearn.base import clone
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
from read_file import read_file
import pdb
import numpy as np
import methods
import os.path
import copy

def evaluate_model_regression(dataset, save_file, random_state, est, 
        hyper_params):

    est_name = type(est).__name__
    print(est_name)
    # load data
    features, labels, feature_names = read_file(dataset)
    # generate train/test split
    # X_train, X_test, y_train, y_test = train_test_split(features, labels,
    #                                                     train_size=0.75,
    #                                                     test_size=0.25,
    #                                                     random_state=None)
    # scale and normalize the data
    dataname = dataset.split('/')[-1][:-7]
    print('dataset:',dataname)
    print('features:',features.shape)
    print('labels:',labels.shape)
    # define CV strategy for hyperparam tuning
    cv = KFold(n_splits=10, shuffle=True,random_state=random_state)

    # Grid Search
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # grid_est.fit(X_train,y_train)
        cv_results = []
        param_grid = list(ParameterGrid(hyper_params))
        #clone estimators
        if est_name != 'hyperopt_estimator':
            Clfs = [clone(est).set_params(**p) for p in param_grid]
        else:
            Clfs = [copy.deepcopy(est).set_params(**p) for p in param_grid]


        for i,clf in enumerate(Clfs):
            print('running',clf.get_params(),'...')
            # fix the seed of the estimator if it has one
            for a in ['random_state','seed']:
                if hasattr(clf,a):
                    setattr(clf,a,random_state)
                    # clf.random_state = random_state
            # get the CV score on the data
            cv_scores = cross_validate(clf, features, labels, 
                    scoring= ['r2','explained_variance',
                        'neg_mean_squared_error','neg_mean_absolute_error'],
                    cv=cv)
            cv_results.append({
                   'parameters': clf.get_params(),
                   'r2_cv_mean': np.mean(cv_scores['test_r2']),
                   'explained_variance_cv_mean': 
                        np.mean(cv_scores['test_explained_variance']),
                   'neg_mean_squared_error_cv_mean': 
                        np.mean(cv_scores['test_neg_mean_squared_error']),
                   'neg_mean_absolute_error_cv_mean':
                        np.mean(cv_scores['test_neg_mean_absolute_error']),
                   'time': np.mean(cv_scores['fit_time'])
                   # 'test_bal_accuracy':test_bal_accuracy
                   })
            # if 'AutoSklearn' in est_name:
            #     print('Autosklearn cv_results: ', clf.cv_results_)
            #     df_as = pd.DataFrame.from_records(data=clf.cv_results_,
            #             columns=clf.cv_results_.keys())
                # df_as.to_csv(save_file.split('.')[0]+'_'
                #         +est_name+'_cv_results.csv',
                #         index=False)

    # print results
    df = pd.DataFrame.from_records(data=cv_results,
            columns=cv_results[0].keys())
    df['seed'] = random_state
    df['dataset'] = dataname
    df['algorithm'] = est_name
    # df['parameters_hash'] = df['parameters'].apply(lambda x:
    #     hash(frozenset(x.items())))
    print('dataframe columns:',df.columns)
    print(df[:10])
    if os.path.isfile(save_file):
        # if exists, append
        df.to_csv(save_file, mode='a', header=False, index=False)
    else:
        df.to_csv(save_file, index=False)



###############################################################################
# main entry point
###############################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="Evaluate a method on a dataset.", 
            add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,
            type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-save_file', action='store', dest='SAVE_FILE',
            default=None, type=str, help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
            default=None, type=int, help='Seed / trial')

    args = parser.parse_args()
    # import algorithm 
    print('import from','methods.'+args.ALG)
    algorithm = importlib.__import__('methods.'+args.ALG,globals(),locals(),
                                   ['est','hyper_params'])

    print('algorithm:',algorithm.est)
    print('hyperparams:',algorithm.hyper_params)
    evaluate_model_regression(args.INPUT_FILE, args.SAVE_FILE, 
            args.RANDOM_STATE, algorithm.est, algorithm.hyper_params)
