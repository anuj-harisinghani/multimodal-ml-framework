from typing import List
import pandas as pd

from classes.handlers.ParamsHandler import ParamsHandler
from sklearn.model_selection import StratifiedKFold
from classes.handlers.DataHandler import DataHandler


class DataSplitter:
    def __init__(self):
        # self.nfolds = nfolds
        # self.data = data
        params = ParamsHandler.load_parameters('settings')
        self.random_seed = params['random_seed']
        self.mode = params['mode']

    # @staticmethod
    def make_splits(self, data: dict, nfolds: int) -> List:
        pass
