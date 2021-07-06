import numpy as np
import pandas as pd
import os

from classes.handlers.ParamsHandler import ParamsHandler


class PIDExtractor:
    def __init__(self):
        pass

    @staticmethod
    def get_list_of_pids(mode: str, task: str, modalities: dict, extraction_method: str, pid_file_path: str):
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
                pids_mod = [np.intersect1d(pids_mod[i], pids_mod[i + 1]) for i in range(len(pids_mod) - 1)]

        # for fusion mode, we require a union of PIDs taken from each modality (which were intersected internally within a modality)
        elif mode == 'fusion':
            while len(pids_mod) > 1:
                pids_mod = [np.union1d(pids_mod[i], pids_mod[i + 1]) for i in range(len(pids_mod) - 1)]

        # intersecting the final list of PIDs with diagnosis, to get the PIDs with valid diagnosis
        pids = list(np.intersect1d(pids_mod[0], pids_diag))

        # saving PIDs to a file
        # not using pickle here since we should be able to view which PIDs have been utilized in an experiment
        '''
        make it csv here
        '''
        pd.DataFrame(pids, columns=['interview']).to_csv(pid_file_path)

        # with open(pid_file_path, 'w') as f:
        #     for pid in pids:
        #         f.write(pid + '\n')

        return pids
