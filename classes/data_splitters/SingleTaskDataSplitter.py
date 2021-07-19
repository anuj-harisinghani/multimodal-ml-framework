from classes.data_splitters.DataSplitter import DataSplitter

from typing import List
import numpy as np
from sklearn.model_selection import StratifiedKFold


class SingleTaskDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, nfolds: int) -> List:
        x = data['x']
        y = data['y']
        labels = np.array(data['labels'])
        fold_data = []

        folds = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=self.random_seed).split(x, y, groups=labels)

        for train_index, test_index in folds:
            fold = {
                'x_train': x.values[train_index],
                'y_train': y.values[train_index],
                'x_test': x.values[test_index],
                'y_test': y.values[test_index],
                'train_labels': labels[train_index],
                'test_labels': labels[test_index]
            }
            fold_data.append(fold)

        return fold_data






