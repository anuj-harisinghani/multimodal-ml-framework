import numpy as np
import pandas as pd
import os

from classes.handlers.ParamsHandler import ParamsHandler


class PIDExtractor:
    def __init__(self, mode: str, extraction_method: str, output_folder: str, pid_file_paths: dict):
        self.mode = mode
        self.extraction_method = extraction_method
        self.output_folder = output_folder
        self.pid_file_paths = pid_file_paths
        self.superset_ids = []

    # find a better name for inner_get_list_of_pids
    def inner_get_list_of_pids(self, task: str) -> list:

        """
        :param task: the task for which PIDs are required
        :return: list of PIDs that satisfy the task and modality constraints
        """

        # get modalities of the particular task
        modalities = ParamsHandler.load_parameters(task)['modalities']

        # if self.extraction_method == 'default':
        #     pass

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
                task_mod_dict = {'CookieTheft': 1, 'Reading': 2, 'Memory': 3}
                task_mod = task_mod_dict[task]

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
        if self.mode == 'single_tasks' or self.mode == 'ensemble':
            while len(pids_mod) > 1:
                pids_mod = [np.intersect1d(pids_mod[i], pids_mod[i + 1]) for i in range(len(pids_mod) - 1)]

        # for fusion mode, we require a union of PIDs taken from each modality (which were intersected internally within a modality)
        elif self.mode == 'fusion':
            while len(pids_mod) > 1:
                pids_mod = [np.union1d(pids_mod[i], pids_mod[i + 1]) for i in range(len(pids_mod) - 1)]

        # intersecting the final list of PIDs with diagnosis, to get the PIDs with valid diagnosis
        pids = list(np.intersect1d(pids_mod[0], pids_diag))

        return pids

    def get_list_of_pids(self, tasks: list):
        superset_ids = []
        for task in tasks:
            # getting pids and saving them at pid_file_path for each task
            pid_file_path = self.pid_file_paths[task]
            pids = self.inner_get_list_of_pids(task=task)
            pd.DataFrame(pids, columns=['interview']).to_csv(pid_file_path)

            superset_ids.append(pids)

        if self.mode == 'fusion':
            # getting superset_ids for fusion
            while (len(superset_ids)) > 1:
                superset_ids = [np.union1d(superset_ids[i], superset_ids[i + 1]) for i in range(len(superset_ids) - 1)]

            self.superset_ids = superset_ids[0]
            super_pids_file_path = os.path.join('assets', self.output_folder, self.extraction_method + '_super_pids.csv')
            print('superset_ids created!')
            pd.DataFrame(self.superset_ids, columns=['interview']).to_csv(super_pids_file_path)


