from abc import ABC

from classes.data_splitters.DataSplitter import DataSplitter


class CrossValidator(ABC):
    def __init__(self, mode:str):
        self.__splitter = DataSplitterFactory().get(mode)
        self.__trainer = TrainersFactory().get(mode)
        pass

    # @staticmethod
    # def cross_validate(self, m: Union[Model, List], t: Union[Task, List]) -> Dict:
    #     pass
    def cross_validate(self, mode: str, tasks_data: dict) -> dict:
        for task in tasks_data.keys():
            for modality, modality_data in tasks_data[task].items():
                splits = self.__splitter.make_split()
                trainer.train(splits)

        return {}