from glob import glob
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit long jobs.",
                                     add_help=False)
    parser.add_argument('DATA_PATH',type=str)
    parser.add_argument('--r',action='store_true',dest='R', default=False)
    parser.add_argument('--c',action='store_true',dest='C', default=False)
    parser.add_argument('-ml',action='store',dest='mls', type=str, 
            default='')
    parser.add_argument('--long',action='store_true',dest='LONG', default=False)
    parser.add_argument('-n_trials',action='store',dest='NTRIALS', default=1)
    parser.add_argument('-results',action='store',dest='RDIR',
            default='../results/regression',type=str,help='Results directory')
    parser.add_argument('-q',action='store',dest='Q',default='',type=str,
            help='Results directory')
    parser.add_argument('-trials',action='store',dest='TRIALS',
            default='',type=str, help='specify specific trials to run')
    parser.add_argument('-dataset',action='store',dest='DATASET',default='',type=str,
            help='run a specific dataset by name')
    args = parser.parse_args()

datapath = args.DATA_PATH 

if args.Q == '':
    if args.LONG:
        q = 'moore_long'
    else:
        q = 'mooreai_normal'
else:
    q = args.Q
lpc_options = '--lsf -q {Q} -m 12000 -n_jobs 1'.format(Q=q)
datasets = '*' if args.DATASET == '' else args.DATASET

if args.TRIALS != '':
    trials = '-trials ' + args.TRIALS
else:
    trials = ''

if args.mls == '':
    if args.C:
        args.mls = (
            'AdaBoostClassifier,BernoulliNB,DecisionTreeClassifier,'
            'ExtraTreesClassifier,GaussianNB,GradientBoostingClassifier,'
            'KNeighborsClassifier,LogisticRegression,'
            'PassiveAggressiveClassifier,RandomForestClassifier,'
            'SGDClassifier,SVC')
    elif args.R:
        args.mls = (
            'DecisionTreeRegressor,RandomForestRegressor,'
            'GradientBoostingRegressor,XGBRegressor,KNeighborsRegressor,'
            'KernelRidge,LassoLarsCV,SVR')

if args.R:
    
    #mls = ','.join([ml + 'R' for ml in args.mls.split(',')])
    mls = args.mls 
    for f in glob(datapath + "/regression/"+datasets+"/*.tsv.gz"):
        jobline =  ('python analyze.py --r {DATA} '
                   '-ml {ML} '
                   '-results {RDIR} -n_trials {NT} {T} {LPC}').format(
                           DATA=f,
                           LPC=lpc_options,
                           ML=mls,
                           RDIR=args.RDIR,
                           NT=args.NTRIALS,
                           T=trials)
        print(jobline)
        os.system(jobline)

if args.C:
    #mls = ','.join([ml + 'C' for ml in args.mls.split(',')])
    mls = args.mls     
    for i,f in enumerate(glob(datapath + "/classification/"+datasets+"/*.tsv.gz")):
#    if i==0:
        jobline =  ('python analyze.py {DATA} '
               '-ml {ML} '
               '-results {RDIR} -n_trials {NT} {T} {LPC}').format(DATA=f,
                                                      LPC=lpc_options,
                                                      ML=mls,
                                                      RDIR=args.RDIR,
                                                      T=trials,
                                                      NT=args.NTRIALS)
        print(jobline)
        os.system(jobline)
