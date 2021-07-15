from classes.data_splitters.DataSplitter import DataSplitter
from typing import List
import pandas as pd

from sklearn.model_selection import StratifiedKFold


class SingleTaskDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, nfolds: int) -> List:
        x = data['x']
        y = data['y']
        labels = data['labels']
        fold_data = []

        folds = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=self.random_seed).split(x, y, groups=labels)

        for train_index, test_index in folds:
            fold = {}
            fold['x_train'] = x.values[train_index]
            fold['y_train'] = y.values[train_index]
            fold['x_test'] = x.values[test_index]
            fold['y_test'] = y.values[test_index]
            fold['train_labels'] = labels.values[train_index]
            fold['test_labels'] = labels.values[test_index]
            fold_data.append(fold)






