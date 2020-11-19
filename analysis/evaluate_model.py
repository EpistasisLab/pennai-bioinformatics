import sys
import itertools
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, train_test_split, ParameterGrid,
cross_val_score)
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import balanced_accuracy_score
from metrics import balanced_accuracy_score
from sklearn.base import clone
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
import pdb
import numpy as np
import methods
import os.path
import copy

def evaluate_model(dataset, save_file, random_state, est, hyper_params):

    est_name = type(est).__name__
    print(est_name)
    # load data
    features, labels, feature_names = read_file(dataset)
    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        train_size=0.75,
                                                        test_size=0.25,
                                                        random_state=None)
    # scale and normalize the data
    dataname = dataset.split('/')[-1][:-7]
    print('dataset:',dataname)
    print('X_train:',X_train.shape)
    print('y_train:',y_train.shape)
    # define CV strategy for hyperparam tuning
    cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=random_state)
    # grid_est = GridSearchCV(est,cv=cv, param_grid=hyper_params,
    #         verbose=1,n_jobs=-1,scoring=balanced_accuracy_score,error_score=0.0)

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
            # get the CV score on the training data
            if est_name not in ['hyperopt_estimator','AutoSklearnClassifier']:
                bal_accuracy = np.mean(cross_val_score(clf,X_train,y_train,cv=cv))
            else:
                bal_accuracy = 0
            # refit the model to all the data
            clf.fit(X_train, y_train)
            # get a holdout test score
            test_bal_accuracy = balanced_accuracy_score(y_test, clf.predict(X_test))
            cv_results.append({
                   'parameters': clf.get_params(),
                   'bal_accuracy':bal_accuracy,
                   'test_bal_accuracy':test_bal_accuracy
                   })
            if 'AutoSklearn' in est_name:
                print('Autosklearn cv_results: ', clf.cv_results_)
                df_as = pd.DataFrame.from_records(data=clf.cv_results_,
                        columns=clf.cv_results_.keys())
                df_as.to_csv(save_file.split('.')[0]+'_'+est_name+'_cv_results.csv',
                        index=False)

    # print results
    df = pd.DataFrame.from_records(data=cv_results,columns=cv_results[0].keys())
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



################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a method on a dataset.", 
                                     add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-save_file', action='store', dest='SAVE_FILE',default=None,
            type=str, help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',default=None,
            type=int, help='Seed / trial')

    args = parser.parse_args()
    # import algorithm 
    print('import from','methods.'+args.ALG)
    algorithm = importlib.__import__('methods.'+args.ALG,globals(),locals(),
                                   ['est','hyper_params'])

    print('algorithm:',algorithm.est)
    print('hyperparams:',algorithm.hyper_params)
    evaluate_model(args.INPUT_FILE, args.SAVE_FILE, args.RANDOM_STATE, 
                   algorithm.est, algorithm.hyper_params)
