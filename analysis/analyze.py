import pandas as pd
import numpy as np
import argparse
import os, errno, sys
from joblib import Parallel, delayed

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="Run Sklearn on PMLB.", add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='LEARNERS',default=None,
            type=str, help='Comma-separated list of ML methods to use (should '
            'correspond to a py file name in learners/)')
    parser.add_argument('--lsf', action='store_true', dest='LSF', default=False, 
            help='Run on an LSF HPC (using bsub commands)')
    parser.add_argument('--r', action='store_true', dest='REGRESSION', 
            default=False, help='run regression')
    parser.add_argument('-metric',action='store', dest='METRIC', default='f1_macro', 
            type=str, help='Metric to compare algorithms')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=1,type=int,
            help='Number of parallel jobs')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,
            type=int, help='Number of parallel jobs')
    parser.add_argument('-label',action='store',dest='LABEL',default='class',
            type=str,help='Name of class label column')
    parser.add_argument('-results',action='store',dest='RDIR',default='results',
            type=str,help='Results directory')
    parser.add_argument('-q',action='store',dest='QUEUE',default='moore_normal',
            type=str,help='LSF queue')
    parser.add_argument('-m',action='store',dest='M',default=4096,type=int,
            help='LSF memory request and limit (MB)')
    parser.add_argument('-trials',action='store',dest='TRIALS',default='',type=str,
            help='specify specific trials to run')
    args = parser.parse_args()
      
    learners = [ml for ml in args.LEARNERS.split(',')]  # learners
    print('learners:',learners)

    model_dir = 'ml'

    dataset = args.INPUT_FILE.split('/')[-1].split('.')[0]
    print('dataset:',dataset)

    results_path = '/'.join([args.RDIR, dataset]) + '/'
    # make the results_path directory if it doesn't exit 
    try:
        os.makedirs(results_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # initialize output files
    for ml in learners:
        #write headers
        save_file = results_path + '/' + dataset + '_' + ml + '.csv'  
        
       
    # Seeds: 100 random seeds were generated using the following lines:
    # np.random.seed(18486)
    # np.random.randint(2**15-1,size=100)
    seeds = [24800,  3470,  8852,  3741, 16723, 31455, 20524, 22432, 19968,
       29112, 29703, 20149, 13518, 27574,  2215, 22491, 19823, 31243,
       32073, 32469,  1393, 13260, 17677,  8317, 13981, 10217, 15242,
       31019,  5638, 10550, 26570, 23548,  1070, 14866, 14152, 13647,
       12145, 25494, 32703, 15792, 21172, 10827,  6231, 27412, 22682,
       16491,   595, 29770, 26088, 13653, 25091,  8409, 29432,  3979,
        3839, 24030, 24118, 11306,  7861,  8717, 16623, 12055, 19274,
        6800,  7814,  3607, 30121, 17126,  4395, 12063, 28893, 21831,
       23483, 12353, 25345, 15616, 21264, 21973,  3747, 30670,  8125,
       32263, 22070, 22705, 10233, 27885, 11064, 16481, 10627, 24258,
       12235, 28616, 22516, 28559,  8418,  3538, 22463,  9575, 28691,
       12256]
    # if -trials is specified, it overrides the N_TRIALS argument
    if args.TRIALS != '':
        trials = [int(t) for t in args.TRIALS.split(',')]
    else:
        trials = range(args.N_TRIALS)
    # write run commands
    all_commands = []
    job_info=[]
    if args.REGRESSION:
        eval_file='evaluate_model_regression'
    else:
        eval_file='evaluate_model'

    for t in trials:
        # random_state = np.random.randint(2**15-1)
        random_state = seeds[t]
        print('random_seed:',random_state)
        
        for ml in learners:
            save_file = results_path + '/' + dataset + '_' + ml + '.csv'  
            
            all_commands.append('python -u {EF}.py '
                                '{DATASET}'
                                ' -ml {ML}'
                                ' -save_file {SAVEFILE}'
                                ' -seed {RS}'.format(EF=eval_file,
                                                     ML=ml,
                                                     DATASET=args.INPUT_FILE,
                                                     SAVEFILE=save_file,
                                                     RS=random_state)
                                )
            job_info.append({'ml':ml,'dataset':dataset,
                'results_path':results_path,'seed':str(random_state)})

    if args.LSF:    # bsub commands
        for i,run_cmd in enumerate(all_commands):
            job_name = (job_info[i]['ml'] + '_' + job_info[i]['dataset'] + '_' +
                job_info[i]['seed'])
            out_file = job_info[i]['results_path'] + job_name + '_%J.out'
            error_file = out_file[:-4] + '.err'
            
            bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} -q {QUEUE} '
                       '-R "span[hosts=1] rusage[mem={M}]" -M {M} ').format(
                               OUT_FILE=out_file,
                               JOB_NAME=job_name,
                               QUEUE=args.QUEUE,
                               N_CORES=args.N_JOBS,
                               M=args.M)
            
            bsub_cmd +=  '"' + run_cmd + '"'
            print(bsub_cmd)
            os.system(bsub_cmd)     # submit jobs 

    else:   # run locally  
        for run_cmd in all_commands: 
            print(run_cmd)
        Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd) 
                for run_cmd in all_commands )
