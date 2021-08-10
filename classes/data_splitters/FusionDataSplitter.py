from classes.data_splitters.DataSplitter import DataSplitter
from classes.handlers.ParamsHandler import ParamsHandler

import numpy as np
import pandas as pd
import os
import random
import copy


class FusionDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, seed: int) -> list:
        self.random_seed = seed
        x = data['x']
        y = data['y']
        labels = np.array(data['labels'])
        fold_data = []

        params = ParamsHandler.load_parameters("settings")
        output_folder = params["output_folder"]
        extraction_method = params["PID_extraction_method"]
        tasks = params["tasks"]

        method = 1
        splits = []
        # option 1: Superset PIDs
        '''
        # for creating Superset_IDs here from the list of pids for each task, getting them from the files saved outside
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
        if method == 1:
            # get list of superset_ids from the saved file
            super_pids_file_path = os.path.join('results', output_folder, extraction_method + '_super_pids.csv')
            Superset_IDs = list(pd.read_csv(super_pids_file_path)['interview'])

            # random shuffle based on random seed
            random.Random(self.random_seed).shuffle(Superset_IDs)
            splits = np.array_split(Superset_IDs, self.nfolds)

        # option 2: Split an intersection of pids across tasks, then split the out-of-intersection pids, then merge them equally
        if method == 2:
            pid_file_paths = {task: os.path.join('results', output_folder, extraction_method + '_' + task + '_pids.csv') for task in tasks}
            pids = [list(pd.read_csv(pid_file_paths[task])['interview']) for task in tasks]

            uni_pids = inter_pids = copy.deepcopy(pids)

            # creating intersection of pids across tasks
            while len(inter_pids) > 1:
                inter_pids = [np.intersect1d(inter_pids[i], inter_pids[i + 1]) for i in range(len(inter_pids) - 1)]
            inter_pids = list(inter_pids[0])

            # creating union of pids across tasks
            while len(uni_pids) > 1:
                uni_pids = [np.union1d(uni_pids[i], uni_pids[i+1]) for i in range(len(uni_pids) - 1)]
            uni_pids = uni_pids[0]

            # difference in uni_pids and inter_pids
            diff_pids = list(np.setxor1d(uni_pids, inter_pids))

            # shuffling before splitting
            random.Random(self.random_seed).shuffle(inter_pids)
            random.Random(self.random_seed).shuffle(diff_pids)

            inter_splits = np.array_split(inter_pids, self.nfolds)
            diff_splits = np.array_split(diff_pids, self.nfolds)

            splits = []
            for i in range(self.nfolds):
                splits.append(np.append(inter_splits[i], diff_splits[i]))

        # option 3: Split all tasks pids seperately, then merge them equally
        '''
        not working properly: inside each task, there's 10 splits
        when combining each split across the tasks, there can be pids which are duplicated in a separate split from a different task
        which causes the final splits to be almost 3 times the size of what a normal split should be.
        to have global "uniqueness" across the splits, a union across all splits must be used which already works in methods 1 and 2.
        so this one is scrapped (or is just kept here for the lulz)
        '''
        if method == 3:
            pid_file_paths = {task: os.path.join('results', output_folder, extraction_method + '_' + task + '_pids.csv') for task in tasks}
            pids = [list(pd.read_csv(pid_file_paths[task])['interview']) for task in tasks]

            split_pids = []
            # splitting each list of pids separately
            for task_pids in pids:
                random.Random(self.random_seed).shuffle(task_pids)
                split_pids.append(np.array_split(task_pids, self.nfolds))

            splits = []
            for j in range(self.nfolds):
                task_pids = [np.union1d(split_pids[i][j], split_pids[i+1][j]) for i in range(len(split_pids)-1)]
                splits.append(np.unique(task_pids[0]))

        # after creating the splits:
        # manually creating folds and filling data
        folds = [np.intersect1d(group, labels) for group in splits]
        for i in range(len(folds)):
            fold = {}

            test = folds[i]
            train = np.concatenate(folds[:i] + folds[i + 1:])

            train_index = [np.where(x.index == train[j])[0][0] for j in range(len(train))]
            test_index = [np.where(x.index == test[j])[0][0] for j in range(len(test))]

            fold['x_train'] = x.values[train_index]
            fold['y_train'] = y.values[train_index]
            fold['x_test'] = x.values[test_index]
            fold['y_test'] = y.values[test_index]
            fold['train_labels'] = labels[train_index]
            fold['test_labels'] = labels[test_index]
            fold_data.append(fold)

        return fold_data
