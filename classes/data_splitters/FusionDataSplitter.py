from classes.data_splitters.DataSplitter import DataSplitter
from classes.handlers.PIDExtractor import PIDExtractor
from classes.handlers.ParamsHandler import ParamsHandler

from typing import List
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold



class FusionDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, nfolds: int) -> List:
        # x = data['x']
        # y = data['y']
        # labels = np.array(data['labels'])
        # fold_data = []

        params = ParamsHandler.load_parameters("settings")
        output_folder = params["output_folder"]
        extraction_method = params["PID_extraction_method"]

        # option 1: Superset PIDs
        '''
        tasks = list(data.keys())
        Superset_IDs = []

        #   # get superset pids from the provided tasks. the pids are already stored in the output_folder directory
        for task in tasks:
            pid_file_path = os.path.join('results', output_folder, extraction_method + '_' + task + '_pids.csv')
            pids = list(pd.read_csv(pid_file_path)['interview'])
            Superset_IDs.append(pids)

        #   # taking a union of all the pids gotten from the different tasks
        while(len(Superset_IDs)) > 1:
            Superset_IDs = [np.union1d(Superset_IDs[i], Superset_IDs[i+1]) for i in range(len(Superset_IDs)-1)]

        Superset_IDs = Superset_IDs[0]
        '''
        super_pids_file_path = os.path.join('results', output_folder, extraction_method + '_super_pids.csv')
        Superset_IDs = list(pd.read_csv(super_pids_file_path)['interview'])


        # option 2: Split an intersection of pids across tasks, then split the out-of-intersection pids, then merge them equally
        # option 3: Split all tasks pids seperately, then merge them equally


        folds = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=self.random_seed).split()


        for train_index, test_index in folds:
            fold = {}
            fold['x_train'] = x.values[train_index]
            fold['y_train'] = y.values[train_index]
            fold['x_test'] = x.values[test_index]
            fold['y_test'] = y.values[test_index]
            fold['train_labels'] = labels[train_index]
            fold['test_labels'] = labels[test_index]
            fold_data.append(fold)

        return fold_data
