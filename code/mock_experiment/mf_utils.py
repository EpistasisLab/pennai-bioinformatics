# hard-coded json metafeature data.
import json
import pdb
import pandas as pd

def update_dataset_mf(dataset_mf, results_data):
        """Grabs metafeatures of datasets in results_data
        """
        dataset_metafeatures = []
        for d in results_data['dataset'].unique():
            if len(dataset_mf)==0 or d not in dataset_mf.index:
                # fetch metafeatures from server for dataset and append
                df = local_get_metafeatures(d)        
                df['dataset'] = d
                # print('metafeatures:',df)
                dataset_metafeatures.append(df)
        if dataset_metafeatures:
            df_mf = pd.concat(dataset_metafeatures).set_index('dataset')
            # print('df_mf:',df_mf['dataset'], df_mf) 
            dataset_mf = dataset_mf.append(df_mf)
            # print('dataset_mf:\n',dataset_mf)
        return dataset_mf

def local_get_metafeatures(d):
        """Fetch dataset metafeatures from file"""

        mf_path = 'mock_experiment/metafeatures/api/datasets/'
        try:
            # req = urllib.request.Request(mf_path+'/'+d, data=params)
            # r = urllib.request.urlopen(req)
           with open(mf_path+'/'+d+'/metafeatures.json') as data_file:    
                   data = json.load(data_file)
            # data = json.loads(r.read().decode(r.info().get_param('charset')
            #                           or 'utf-8'))[0]
        except Exception as e:
            print('exception when grabbing metafeature data for',d)
            raise e
        
        df = pd.DataFrame.from_records(data,columns=data.keys(),index=[0])
        df['dataset'] = d
        df.sort_index(axis=1, inplace=True)

        # print('df:',df)
        return df
