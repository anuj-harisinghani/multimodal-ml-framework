from classes.handlers.ParamsHandler import ParamsHandler

from typing import List
import pandas as pd
from scipy import stats
import numpy as np

'''
to be called inside trainers
'''


class FeatureSelector:
    def __init__(self):
        params = ParamsHandler.load_parameters('settings')
        self.fs_pairwise = params['fs_pairwise_correlation']
        self.fs_outcome = params['fs_outcome_correlation']

    def select_features(self, fold_data: dict, x_columns: list) -> tuple:
        # extract the data from fold_data
        x_train = fold_data['x_train']
        x_test = fold_data['x_test']
        y_train = fold_data['y_train'].ravel()

        # getting list of features (column names) from the x data
        feature_names = x_columns

        # choosing which features to remove based on the pairwise correlation coefficient
        to_drop = self.get_pairwise_correlated_features(x=x_train)
        x_train = np.delete(x_train, to_drop, axis=1)
        x_test = np.delete(x_test, to_drop, axis=1)

        if feature_names is not None:
            for idx in sorted(to_drop, reverse=True):
                del feature_names[idx]

        # choosing which features to keep based on a correlation cutoff threshold
        # method = 'pearson'
        indices = self.get_top_correlation_features_by_cutoff(x_train, y_train)
        nfeats = len(indices)
        if nfeats == 0:
            raise ValueError("No features left after feature selection!")

        # making the feature_selected data
        x_train_fs = x_train[:, indices]
        x_test_fs = x_test[:, indices]
        selected_feature_names = [feature_names[idx] for idx in indices]

        return x_train_fs, x_test_fs, selected_feature_names

    # function to get list of features to remove based on pairwise correlation
    def get_pairwise_correlated_features(self, x: np.ndarray):
        df = pd.DataFrame(x)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.fs_pairwise)]
        return to_drop

    # function to get list of features to keep based on pearson correlation over a certain threshold
    def get_top_correlation_features_by_cutoff(self, x: np.ndarray, y: np.ndarray, method: str = 'pearson'):
        if method == 'pearson':
            return [column for column in range(x.shape[1]) if abs(stats.pearsonr(y, x[:, column])[0]) > self.fs_outcome]
        elif method == 'pointbiserial':
            return [column for column in range(x.shape[1]) if abs(stats.pointbiserialr(y, x[:, column])[0]) > self.fs_outcome]
        else:
            raise KeyError("Invalid correlation type!")
