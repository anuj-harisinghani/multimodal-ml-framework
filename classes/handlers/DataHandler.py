# noinspection PyUnusedLocal
from typing import List

import numpy as np
import pandas as pd
import os

from classes.handlers.ParamsHandler import ParamsHandler


class DataHandler:

    def __init__(self, mode: str, extraction_method: str, output_folder: str):
        self.extraction_method = extraction_method
        self.output_folder = output_folder
        self.pid_file_path = os.path.join('results', output_folder, extraction_method + '_pids.txt')
        self.mode = mode


    def load_data(self, tasks: List) -> List:
        for task in tasks:
            params = ParamsHandler.load_parameters(task)
            modalities = params['modalities']
            features = params['features']
            feature_set = params['feature_sets']

            # old way: when we had pids extracted in here and passed to get_data
            # pids = self.get_list_of_pids(mode, modalities, task)
            # data = self.get_data(task, feature_set, pids)

            # new way: calling pid extraction here, saving pids to a file, then making get_data get the pids from the file
            self.get_list_of_pids(self.mode, modalities, task, self.extraction_method, self.pid_file_path)
            X, y, labels = self.get_data(task, feature_set, self.pid_file_path)

        # not sure if the modes are of any use here, for loading data
        if self.mode == "single_tasks":
            pass
        if self.mode == "fusion":
            pass
        if self.mode == "ensemble":
            pass

        # probably should just return X, y, labels here and let cross_validator work with it
        return []
    '''
    put this method in a separate class
    '''
    @staticmethod
    def get_list_of_pids(mode: str, modalities: dict, task: str, extraction_method: str, pid_file_path: str) -> List:
        """
        :param mode: mode specifies how the PIDs should be handled (single = intersect everything, fusion = union of modality level PIDs, intersect within modality)
        :param modalities: which modalities (based on the task) would influence selected PIDs
        :param task: the task for which PIDs are required
        :param extraction_method: the method by which PIDs should be extracted, specified by user in the "settings" file
        :param pid_file_path: the path at which the list of PIDs will be created
        :return: list of PIDs that satisfy the task and modality constraints
        """

        if extraction_method == 'default':
            pass
        data_path = os.path.join('datasets', 'csv_tables')
        database = ParamsHandler.load_parameters('database')
        modality_wise_datasets = database['modality_wise_datasets']
        plog_threshold = ParamsHandler.load_parameters('settings')['eye_tracking_calibration_flag']

        pids_diag = pd.read_csv(os.path.join(data_path, 'diagnosis.csv'))['interview']

        pids_mod = []
        for modality in modalities:
            task_mod = task
            filename = modality_wise_datasets[modality]

            # for eye modality, PIDs from eye_fixation and participant_log are intersected
            if modality == 'eye':
                table_eye = pd.read_csv(os.path.join(data_path, filename[0]))
                pids_eye = table_eye.loc[table_eye['task'] == task_mod]['interview']

                table_plog = pd.read_csv(os.path.join(data_path, filename[1]))
                pids_plog = table_plog[table_plog['Eye-Tracking Calibration?'] >= plog_threshold]['interview']

                pids_mod.append(np.intersect1d(pids_eye, pids_plog))

            # for speech modality the files being accessed (text and audio) have tasks as 1, 2, 3 under the tasks column
            # then PIDs from text and audio are intersected
            if modality == 'speech':
                task_mod = 1 if task == 'CookieTheft' else 2 if task == 'Reading' else 3 if task == 'Memory' else None

                table_audio = pd.read_csv(os.path.join(data_path, filename[0]))  # add support for more than one filenames like for speech
                pids_audio = table_audio.loc[table_audio['task'] == task_mod]['interview']

                table_text = pd.read_csv(os.path.join(data_path, filename[1]))
                pids_text = table_text[table_text['task'] == task_mod]['interview']

                pids_mod.append(np.intersect1d(pids_audio, pids_text))

            # PIDs from moca are used
            if modality == 'moca':
                table_moca = pd.read_csv(os.path.join(data_path, filename[0]))
                pids_moca = table_moca['interview']
                pids_mod.append(pids_moca)

            # PIDs from mm_overall are used
            if modality == 'multimodal':
                table_multimodal = pd.read_csv(os.path.join(data_path, filename[0]))
                pids_multimodal = table_multimodal.loc[table_multimodal['task'] == task_mod]['interview']
                pids_mod.append(pids_multimodal)

        # for single task mode, we require an intersection of all PIDs, from all modalities
        if mode == 'single_tasks':
            while len(pids_mod) > 1:
                pids_mod = [np.intersect1d(pids_mod[i], pids_mod[i+1]) for i in range(len(pids_mod) - 1)]

        # for fusion mode, we require a union of PIDs taken from each modality (which were intersected internally within a modality)
        elif mode == 'fusion':
            while len(pids_mod) > 1:
                pids_mod = [np.union1d(pids_mod[i], pids_mod[i+1]) for i in range(len(pids_mod) - 1)]

        # intersecting the final list of PIDs with diagnosis, to get the PIDs with valid diagnosis
        pids = list(np.intersect1d(pids_mod[0], pids_diag))

        # saving PIDs to a file
        # not using pickle here since we should be able to view which PIDs have been utilized in an experiment
        '''
        make it csv here
        '''
        with open(pid_file_path, 'w') as f:
            for pid in pids:
                f.write(pid + '\n')

        return pids


    @staticmethod
    def get_data(task: str, feature_set: dict, pid_file_path: str) -> tuple:
        feature_path = os.path.join('feature_sets')
        feature_subsets_path = os.path.join(feature_path, 'feature_subsets')
        data_path = os.path.join('datasets', 'csv_tables')

        # get pids from a saved file, which was created by get_list_of_pids based on the conditions given to it
        with open(pid_file_path, 'r') as f:
            pids = [line.rstrip('\n') for line in f]

        # initializing the dataset as the list of PIDs
        dataset = pd.DataFrame(pids, columns=['interview'])
        final_features = []

        # checks if there are multiple features in a list under either eye or language (like there is Audio and Text under lang for Memory)
        features = list(feature_set.values())

        # unpacking all features from their feature sets into final_features
        for feat in features:
            features_subset = ParamsHandler.load_parameters(os.path.join(feature_path, feat))
            final_features.extend(features_subset)

        # getting the list of features that are to be selected from the datasets
        to_select = []
        for feat in final_features:
            # currently cannot get feature_subset for ET based features since they're in different files. but can do the rest of language based features
            # this would require copying feature names from the ET dataset files and putting them in feature_subset files like the lang ones
            # to make this portion work for everything, the Eye based datasets should be combined

            # language based to_select feature_subset would exist in a merged version of Text and Audio datasets.

            to_select.extend(ParamsHandler.load_parameters(os.path.join(feature_subsets_path, feat)))

            # after getting to_select values, get the data from dataset files
            # we could first get the data then merge them, or merge them first then get the data, whichever would be less time consuming

            # if to_select has any features in it, then merge text and acoustic datasets, then choose the columns in the merged set that corresponds to
            # the features in to_select, for the task specified in the beginning

            # def function_here():

            if len(to_select) > 0:
                # for eye features
                # to_select is not necessary to have here for eye based features
                # so this if else statement can be put before we check for to_select, and eye based features can be left out of to_select
                if feat.startswith('eye') or feat == 'fraser':
                    eye_data = pd.read_csv(os.path.join(data_path, feat+'.csv'))
                    eye_dataset = eye_data.loc[eye_data['task'] == task]

                    to_select.append('interview')
                    eye_dataset = eye_dataset[to_select]
                    dataset = pd.merge(dataset, eye_dataset, on='interview')
                    # continue

                # for NLP features
                else:
                    task_mod = 1 if task == 'CookieTheft' else 2 if task == 'Reading' else 3 if task == 'Memory' else None

                    text_data = pd.read_csv(os.path.join(data_path, 'text.csv'))
                    acoustic_data = pd.read_csv(os.path.join(data_path, 'acoustic.csv'))

                    lang_merged = pd.merge(text_data, acoustic_data, on=['interview', 'task'])
                    lang_dataset = lang_merged.loc[lang_merged['task'] == task_mod]

                    to_select.append('interview')
                    lang_dataset = lang_dataset[to_select]
                    dataset = pd.merge(dataset, lang_dataset, on='interview')

        # merging with diagnosis to get patients with valid diagnosis
        diagnosis_data = pd.read_csv(os.path.join(data_path, 'diagnosis.csv'))
        dataset = pd.merge(dataset, diagnosis_data, on='interview')

        # random sample
        dataset.sample(frac=1, random_state=10)

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

        return x, y, labels
