from classes.data_splitters.DataSplitter import DataSplitter
from typing import List
import pandas as pd

class DataSplitterSingleTask(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(data: pd.DataFrame, folds: list) -> List:
        pass
