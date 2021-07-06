from classes.cv.CrossValidator import CrossValidator
from classes.cv.Trainer import Trainer


class SingleModelTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train(self, folds: list):
        pass