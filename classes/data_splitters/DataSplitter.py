from typing import List
import pandas as pd

from classes.handlers.ParamsHandler import ParamsHandler
from sklearn.model_selection import StratifiedKFold


class DataSplitter:
    def __init__(self):
        # self.nfolds = nfolds
        # self.data = data
        params = ParamsHandler.load_parameters('settings')
        self.random_seed = params['random_seed']

    # @staticmethod
    def make_splits(self, data: dict, nfolds: int) -> List:
        pass
