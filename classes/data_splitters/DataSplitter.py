from typing import List
import pandas as pd

from sklearn.model_selection import StratifiedKFold


class DataSplitter:
    def __init__(self, nfolds, data):
        self.nfolds = nfolds
        self.data = data

    # @staticmethod
    def make_splits(self, data: pd.DataFrame, folds: list) -> List:
        pass
