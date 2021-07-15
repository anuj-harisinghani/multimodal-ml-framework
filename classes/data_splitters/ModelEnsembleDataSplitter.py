from classes.data_splitters.DataSplitter import DataSplitter
from typing import List
import pandas as pd


class ModelEnsembleDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, nfolds: int) -> List:
        pass
