from classes.data_splitters.DataSplitter import DataSplitter
from classes.data_splitters.SingleTaskDataSplitter import SingleTaskDataSplitter
from classes.data_splitters.TaskFusionDataSplitter import TaskFusionDataSplitter
from classes.data_splitters.ModelEnsembleDataSplitter import ModelEnsembleDataSplitter


class DataSplitterFactory:
    def __init__(self):
        self.__data_splitters = {
            "single_tasks": SingleTaskDataSplitter(),
            "task_fusion": TaskFusionDataSplitter(),
            "model_ensemble": ModelEnsembleDataSplitter()
        }

    def get(self, mode: str) -> DataSplitter:
        if mode not in self.__data_splitters.keys():
            raise ValueError("Data splitter '{}' not supported! Supported splitters are: {}"
                             .format(mode, self.__data_splitters.keys()))

        return self.__data_splitters[mode]
