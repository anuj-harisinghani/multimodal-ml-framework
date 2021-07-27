from classes.trainer.Trainer import Trainer
from classes.cv.FeatureSelector import FeatureSelector
from classes.handlers.ModelsHandler import ModelsHandler


class SingleModelTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train(self, splits: list, models: list, x_columns: list):

        for model in models:
            for idx, fold in enumerate(splits):
                x_train, y_train = fold['x_train'], fold['y_train'].ravel()
                x_test, y_test = fold['x_test'], fold['y_test'].ravel()
                labels_train, labels_test = fold['train_labels'], fold['test_labels']

                # getting feature selected x_train, x_test and the list of selected features
                x_train_fs, x_test_fs, selected_feature_names = FeatureSelector().select_features(fold_data=fold, x_columns=x_columns)

                # fit the model
                model.fit(x_train_fs, y_train)
