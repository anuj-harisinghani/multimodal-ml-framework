from classes.trainer.Trainer import Trainer
from classes.trainer.SingleModelTrainer import SingleModelTrainer
from classes.trainer.TaskFusionTrainer import TaskFusionTrainer
from classes.trainer.ModelEnsembleTrainer import ModelEnsembleTrainer


class TrainersFactory:
    def __init__(self):
        self.__trainers = {
            "single_tasks": SingleModelTrainer,
            "fusion": TaskFusionTrainer,
            "ensemble": ModelEnsembleTrainer
        }

    def get(self, mode: str) -> Trainer:
        """
        get -> returns a Trainer class from the given mode
        :param mode: single_tasks, fusion or ensemble. Used to choose the type of Trainer.
        :return: Trainer class
        """
        if mode not in self.__trainers.keys():
            raise ValueError("Trainer '{}' not supported! Supported trainers are: {}"
                             .format(mode, self.__trainers.keys()))

        return self.__trainers[mode]()
