# noinspection PyUnusedLocal
from typing import List

import numpy as np
import pandas as pd
import os

from classes.handlers.ParamsHandler import ParamsHandler
from classes.handlers.PIDExtractor import PIDExtractor


class DataHandler:

    def __init__(self, mode: str, output_folder: str, extraction_method: str):
        self.mode = mode
        self.output_folder = output_folder
        self.extraction_method = extraction_method
        self.pid_file_paths = None

    def load_data(self, tasks: List) -> dict:
        tasks_data = {task: None for task in tasks}
        self.pid_file_paths = {task: os.path.join('results', self.output_folder, self.extraction_method + '_' + task + '_pids.csv')
                               for task in tasks}
        PIDExtractor(mode=self.mode, extraction_method=self.extraction_method, output_folder = self.output_folder,
                     pid_file_paths=self.pid_file_paths).get_list_of_pids(tasks=tasks)

        for task in tasks:
            print(task)
            params = ParamsHandler.load_parameters(task)
            modalities = params['modalities']
            features = params['features']
            feature_set = params['feature_sets']

            modality_data = {modality: None for modality in modalities}
            for modality in modalities:
                modality_data[modality] = self.get_data(task, modality, feature_set, self.pid_file_paths[task])
                tasks_data[task] = modality_data

        '''
        tasks_data = {task: None for task in tasks}
        if self.mode == 'single_tasks':
            for task in tasks:
                print(task)
                params = ParamsHandler.load_parameters(task)
                modalities = params['modalities']
                features = params['features']
                feature_set = params['feature_sets']

                self.pid_file_path = os.path.join('results', self.output_folder, self.extraction_method + '_' + task + '_pids.txt')
                PIDExtractor(mode=self.mode, pid_file_path=self.pid_file_path).get_list_of_pids(tasks=task)

                modality_data = {modality: None for modality in modalities}
                for modality in modalities:
                    modality_data[modality] = self.get_data(task, modality, feature_set, self.pid_file_path)
                    tasks_data[task] = modality_data

        elif self.mode == 'fusion':
            tasks_data = {task: None for task in tasks}

            self.pid_file_path = os.path.join('results', self.output_folder, self.mode + '_' + self.extraction_method + '_pids.txt')
            PIDExtractor(mode=self.mode, pid_file_path=self.pid_file_path).get_list_of_pids(tasks=tasks)

            for task in tasks:
                print(task)
                params = ParamsHandler.load_parameters(task)
                modalities = params['modalities']
                features = params['features']
                feature_set = params['feature_sets']

                modality_data = {modality: None for modality in modalities}
                for modality in modalities:
                    modality_data[modality] = self.get_data(task, modality, feature_set, self.pid_file_path)
                    tasks_data[task] = modality_data
        '''


        '''
        tasks_data = {task: None for task in tasks}
        for task in tasks:
            print(task)
            params = ParamsHandler.load_parameters(task)
            modalities = params['modalities']
            features = params['features']
            feature_set = params['feature_sets']

            # old way: when we had pids extracted in here and passed to get_data
            # pids = self.get_list_of_pids(mode, modalities, task)
            # data = self.get_data(task, feature_set, pids)

            # tasks_data = {task: {modality: None for modality in modalities} for task in tasks}
            modality_data = {modality: None for modality in modalities}


            # pid_file_path = os.path.join('results', output_folder, extraction_method + '_pids.txt')
            # pids = get_list_of_pids(mode, task, modalities, extraction_method, pid_file_path)
            for modality in modalities:
                modality_data[modality] = self.get_data(task, modality, feature_set, self.pid_file_path)
                tasks_data[task] = modality_data

            # modality_data = {modality: {'x': x, 'y': y, 'labels': labels} for modality in modalities}

        # not sure if the modes are of any use here, for loading data
        # if self.mode == "single_tasks":
        #     return tasks_data
        # if self.mode == "fusion":
        #     pass
        # if self.mode == "ensemble":
        #     pass

        '''
        return tasks_data


    @staticmethod
    def get_data(task: str, modality: str, feature_set: dict, pid_file_path: str) -> dict:
        feature_path = os.path.join('feature_sets')
        feature_subsets_path = os.path.join(feature_path, 'feature_subsets')
        data_path = os.path.join('datasets', 'csv_tables')

        # get pids from a saved file, which was created by get_list_of_pids based on the conditions given to it
        # with open(pid_file_path, 'r') as f:
        #     pids = [line.rstrip('\n') for line in f]
        pids = pd.read_csv(pid_file_path)

        # initializing the dataset as the list of PIDs
        dataset = pids
        final_features = []
        features = list(feature_set.values())

        # unpacking all features from their feature sets into final_features
        for feat in features:
            features_subset = ParamsHandler.load_parameters(os.path.join(feature_path, feat))
            final_features.extend(features_subset)

        if modality == 'eye':
            for feat in final_features:
                to_select = ['interview']
                if feat.startswith('eye'):
                    print("--", feat)
                    to_select.extend(ParamsHandler.load_parameters(os.path.join(feature_subsets_path, feat)))
                    eye_data = pd.read_csv(os.path.join(data_path, feat + '.csv'))
                    eye_dataset = eye_data.loc[eye_data['task'] == task]

                    eye_dataset = eye_dataset[to_select]
                    dataset = pd.merge(dataset, eye_dataset, on='interview')

        elif modality == 'speech':
            task_mod = 1 if task == 'CookieTheft' else 2 if task == 'Reading' else 3 if task == 'Memory' else None

            # NLP data files merging. No need to put it in the loop as that adds time
            text_data = pd.read_csv(os.path.join(data_path, 'text.csv'))
            acoustic_data = pd.read_csv(os.path.join(data_path, 'acoustic.csv'))

            lang_merged = pd.merge(text_data, acoustic_data, on=['interview', 'task'])

            for feat in final_features:
                to_select = ['interview']
                if not feat.startswith('eye'):
                    print("--", feat)
                    to_select.extend(ParamsHandler.load_parameters(os.path.join(feature_subsets_path, feat)))

                    if feat == 'fraser':
                        fraser_data = pd.read_csv(os.path.join(data_path, feat + '.csv'))
                        fraser_dataset = fraser_data.loc[fraser_data['task'] == task]

                        fraser_dataset = fraser_dataset[to_select]
                        dataset = pd.merge(dataset, fraser_dataset, on='interview')
                        continue

                    lang_dataset = lang_merged.loc[lang_merged['task'] == task_mod]
                    lang_dataset = lang_dataset[to_select]
                    dataset = pd.merge(dataset, lang_dataset, on='interview')

        # to_select = ['interview']

        # for feat in final_features:
        #     # currently cannot get feature_subset for ET based features since they're in different files. but can do the rest of language based features
        #     # this would require copying feature names from the ET dataset files and putting them in feature_subset files like the lang ones
        #     # to make this portion work for everything, the Eye based datasets should be combined
        #
        #     # language based to_select feature_subset would exist in a merged version of Text and Audio datasets.
        #     print(feat)
        #     to_select.extend(ParamsHandler.load_parameters(os.path.join(feature_subsets_path, feat)))
        #
        #     # after getting to_select values, get the data from dataset files
        #     # we could first get the data then merge them, or merge them first then get the data, whichever would be less time consuming
        #
        #     # if to_select has any features in it, then merge text and acoustic datasets, then choose the columns in the merged set that corresponds to
        #     # the features in to_select, for the task specified in the beginning
        #
        #     # def function_here():
        #
        #     if len(to_select) > 0:
        #         if feat.startswith('eye') or feat == 'fraser':
        #             eye_data = pd.read_csv(os.path.join(data_path, feat+'.csv'))
        #             eye_dataset = eye_data.loc[eye_data['task'] == task]
        #
        #             eye_dataset = eye_dataset[to_select]
        #             dataset = pd.merge(dataset, eye_dataset, on='interview')
        #             # continue
        #
        #         # for NLP features
        #         else:
        #             task_mod = 1 if task == 'CookieTheft' else 2 if task == 'Reading' else 3 if task == 'Memory' else None
        #
        #             lang_dataset = lang_merged.loc[lang_merged['task'] == task_mod]
        #             lang_dataset = lang_dataset[to_select]
        #             dataset = pd.merge(dataset, lang_dataset, on='interview')
        #
        #     to_select = ['interview']

        # merging with diagnosis to get patients with valid diagnosis
        diagnosis_data = pd.read_csv(os.path.join(data_path, 'diagnosis.csv'))
        dataset = pd.merge(dataset, diagnosis_data, on='interview')

        # random sample
        dataset = dataset.sample(frac=1, random_state=10)

        # labels
        labels = list(dataset['interview'])

        # y
        y = dataset['diagnosis'] != 'HC'

        # X
        drop = ['interview', 'diagnosis']
        x = dataset.drop(drop, axis=1, errors='ignore')
        x = x.apply(pd.to_numeric, errors='ignore')

        x.index = labels
        y.index = labels

        # return x, y, labels
        return {'x': x, 'y': y, 'labels': labels}
