from abc import ABC

from classes.data_splitters.DataSplitter import DataSplitter
from classes.factories.DataSplitterFactory import DataSplitterFactory
from classes.factories.TrainersFactory import TrainersFactory

from classes.handlers.ParamsHandler import ParamsHandler
from classes.handlers.PIDExtractor import PIDExtractor


class CrossValidator(ABC):
    def __init__(self, mode: str, models: list):
        self.__splitter = DataSplitterFactory().get(mode)
        self.__trainer = TrainersFactory().get(mode)
        self.mode = mode
        self.models = models

    # @staticmethod
    # def cross_validate(self, m: Union[Model, List], t: Union[Task, List]) -> Dict:
    #     pass
    def cross_validate(self, tasks_data: dict) -> dict:
        params = ParamsHandler.load_parameters('settings')
        nfolds = params["folds"]

        if self.mode == 'single_tasks':
            for task in tasks_data.keys():
                for modality, modality_data in tasks_data[task].items():
                    splits, x_columns = self.__splitter.make_splits(data=modality_data, nfolds=nfolds)
                    trained_model = self.__trainer.train(splits, self.models, x_columns)

        # currently this is the same as single_tasks, checking if it would make a difference to keep them separate or not
        elif self.mode == 'fusion':
            for task in tasks_data.keys():
                for modality, modality_data in tasks_data[task].items():
                    splits, x_columns = self.__splitter.make_splits(data=modality_data, nfolds=nfolds)
                    trained_model = self.__trainer.train(splits, x_columns)

        return {}
