import pandas as pd
import numpy as np
import seaborn as sb
from glob import glob
import os
import gc
import scipy.stats
import pdb
from tqdm import tqdm
# iters = 1000
# # n_inits = [1,100,10000]
# n_inits = [10000, 100, 1]
# # nrecs= [1,50,100]
# nrecs = [100,50,1]
# # nrecs=[50]
# # nrecs= [50]
# # nrecs= [100]
# resdir = 'results_pmlb2/'
# knowledgebase='pmlb_sklearn'
# knowledgebase='hibachi'
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Postprocessses mock experiment results")
    parser.add_argument('-n_init',action='store',dest='n_init',type=int,default=1)
    parser.add_argument('-n_recs',action='store',dest='n_recs',type=int,default=1)
    parser.add_argument('-iters',action='store',dest='iters',type=int,default=1)
    parser.add_argument('-batch_size',action='store',dest='batch_size',type=int,default=1000)
    parser.add_argument('-resdir',action='store',dest='RESDIR',type=str,
                        default='results_pmlb2',
                        help='results directory')
    parser.add_argument('-knowledgebase',action='store',dest='KNOWLEDGEBASE',type=str,
                        default='pmlb_sklearn',
                        help='results directory')
    args = parser.parse_args()

    n_init = args.n_init
    n_recs = args.n_recs
    iters = args.iters
    resdir = args.RESDIR
    knowledgebase = args.KNOWLEDGEBASE
    batch_size = args.batch_size

    if not os.path.exists(resdir+'summary/'):
        os.mkdir(resdir+'summary/')
    frames = []
    print(50*'=','\n','n_init:',n_init,'n_recs:',n_recs,'\n',50*'=')
    experiment = (resdir +'experiment_'+knowledgebase+'*_ninit-'+str(n_init)+'_*nrecs-'+str(n_recs)+
                  '_*iters-'+str(iters)+'*csv')
    print(experiment)
    savename = experiment.split('/')[-1].replace('*','').split('csv')[0]
    print('savename:',savename)
#         batch_size = max(round(len(glob(experiment))/10),100)
    print('experiment size:',len(glob(experiment)))
    print('batch_size:',batch_size)
    batch = batch_size 
    df_exp = pd.DataFrame()
    for i,f in tqdm(enumerate(glob(experiment))):
#             print(f)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print('error reading',f)
            print(e)
            continue
#             df = df[['n_recs','n_init','iteration','delta_bal_accuracy','recommender','ranking','iters',
#                     'ml-rec','score-rec','bal_accuracy','best_algorithm','trial']]
        # store dataframe of mean, median, 
        frames.append(df)
        if i > batch or i == len(glob(experiment))-1:
            print('writing batch','batch=',batch)
            batch += batch_size
            df_exp_batch = pd.concat(frames)
            df_exp = df_exp.append(df_exp_batch)
            del df_exp_batch
            frames = []
        gc.collect()
#         pdb.set_trace()
    print('loaded',len(df_exp),'experiments')
    print('columns:',df_exp.columns)
    print('recommenders:',df_exp.recommender.unique())
    print('trials:',len(df_exp.trial.unique()))
    print('datasets:',len(df_exp.dataset.unique()))
    ##########
    # summary stats
    ##########
    print('calculating summary stats')
    df_sum = pd.DataFrame()
#         for c in ['delta_bal_accuracy','ranking']:
    for c in ['delta_bal_accuracy']:
        df_sum[c+'_mean'] = df_exp.groupby(['iteration','recommender'])[c].mean()
        df_sum[c+'_median'] = df_exp.groupby(['iteration','recommender'])[c].median()
        df_sum[c+'_std'] = df_exp.groupby(['iteration','recommender'])[c].std()
        df_sum[c+'_min'] = df_exp.groupby(['iteration','recommender'])[c].min()
        df_sum[c+'_se'] = df_exp.groupby(['iteration','recommender'])[c].apply(scipy.stats.sem)
        df_sum[c+'_q1'] = df_exp.groupby(['iteration','recommender'])[c].quantile(0.25)
        df_sum[c+'_q3'] = df_exp.groupby(['iteration','recommender'])[c].quantile(0.75)
    
    df_sum['iters'] = iters
    df_sum['n_recs'] = n_recs
    df_sum['n_init'] = n_init
    df_sum.drop_duplicates(inplace=True)
    print('saving summary stats to',resdir +savename+'_summary.csv')
    df_sum.to_csv(resdir +'summary/'+savename+'_summary.csv')
    
    ml_options = df_exp['ml-rec'].unique()
    #############
    # save counts of each ML recommendation made by each recommender 
    #############
    print('save counts of each ML recommendation made by each recommender')
    heat_frames = []
    for (it,rec),dfg in tqdm(df_exp.groupby(['iteration','recommender'])):
        df = {} 
        df['iteration'] = it
        df['recommender'] = rec
        for option in ml_options:
            df[option] = 0
        for ml_rec in dfg['ml-rec']:
            df[ml_rec] += 1
        heat_frames.append(df)
#         # add "best" as a recommender
#         for it,dfg in df_exp.groupby(['iteration']):
#             df = {} 
#             df['iteration'] = it
#             df['recommender'] = 'best'
#             for option in ml_options:
#                 df[option] = 0
#             for ml_rec in dfg['best_algorithm']:
#                 for mlr in ml_rec.split('|'):
#                     df[mlr] += 1
#             heat_frames.append(df)
    df_heatmap = pd.DataFrame.from_records(heat_frames).set_index('iteration')
    print('saving',resdir +'summary/'+savename+'_heatmap.csv')
    df_heatmap.to_csv(resdir +'summary/'+savename+'_heatmap.csv')
#         pdb.set_trace()
    gc.collect()
    
    print('done!')
