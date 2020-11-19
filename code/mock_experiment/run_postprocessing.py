import os

n_inits = [10000, 100, 1]
# nrecs = [100,10,1]
nrecs = [10]
iters = 1000
batch_size = 1000
resdir = 'results_pmlb2_r2/'
knowledgebase='pmlb_sklearn'

for ni in n_inits:
    for nr in nrecs:
        os.system('python postprocess_mock_experiment_results.py '
                '-n_init {NI} -n_recs {NR} -iters {IT} -resdir {RD} '
                '-batch_size {BS} -knowledgebase {KB}'.format(
                    NI=ni,
                    NR=nr,
                    IT=iters,
                    RD=resdir,
                    BS=batch_size,
                    KB=knowledgebase)
                )
