from classes.trainer.Trainer import Trainer
from classes.trainer.SingleModelTrainer import SingleModelTrainer
from classes.trainer.TaskFusionTrainer import TaskFusionTrainer
from classes.trainer.ModelEnsembleTrainer import ModelEnsembleTrainer


class TrainersFactory:
    def __init__(self):
        self.__data_splitters = {
            "single_tasks": SingleModelTrainer,
            "fusion": TaskFusionTrainer,
            "ensemble": ModelEnsembleTrainer
        }

    def get(self, mode: str) -> Trainer:
        if mode not in self.__data_splitters.keys():
            raise ValueError("Trainer '{}' not supported! Supported trainers are: {}"
                             .format(mode, self.__data_splitters.keys()))

        return self.__data_splitters[mode]()
