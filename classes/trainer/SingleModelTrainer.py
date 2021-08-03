from classes.trainer.Trainer import Trainer
from classes.cv.FeatureSelector import FeatureSelector
from classes.handlers.ModelsHandler import ModelsHandler

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import numpy as np


class SingleModelTrainer(Trainer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_save_results(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                             acc_saved=None, fms_saved=None, roc_saved=None,
                             precision_saved=None, recall_saved=None, spec_saved=None):
        if precision_saved is None:
            precision_saved = []
        if spec_saved is None:
            spec_saved = []
        if recall_saved is None:
            recall_saved = []
        if roc_saved is None:
            roc_saved = []
        if fms_saved is None:
            fms_saved = []
        if acc_saved is None:
            acc_saved = []

        acc_saved.append(accuracy_score(y_true, y_pred))
        fms_saved.append(f1_score(y_true, y_pred))
        roc_saved.append(roc_auc_score(y_true, y_prob))
        precision_saved.append(precision_score(y_true, y_pred))
        recall_saved.append(recall_score(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec_saved.append(tn / (tn + fp))

        return acc_saved, fms_saved, roc_saved, precision_saved, recall_saved, spec_saved

    def save_results(self, method: str = 'default', acc=None, fms=None, roc=None,
                     precision=None, recall=None, specificity=None,
                     pred=None, pred_prob=None, k_range=None):
        self.results[method] = {"acc": np.asarray(acc),
                                "fms": np.asarray(fms),
                                "roc": np.asarray(roc),
                                "precision": np.asarray(precision),
                                "recall": np.asarray(recall),
                                "specificity": np.asarray(specificity)
                                }

        self.best_k[method] = {
            "acc": np.array(k_range)[np.argmax(np.nanmean(acc, axis=0))],
            "fms": np.array(k_range)[np.argmax(np.nanmean(fms, axis=0))],
            "roc": np.array(k_range)[np.argmax(np.nanmean(roc, axis=0))],
            "precision": np.array(k_range)[np.argmax(np.nanmean(precision, axis=0))],
            "recall": np.array(k_range)[np.argmax(np.nanmean(recall, axis=0))],
            "specificity": np.array(k_range)[np.argmax(np.nanmean(specificity, axis=0))],
            "k_range": k_range}

        self.best_score[method] = {"acc": np.max(np.nanmean(acc, axis=0)),
                                   "fms": np.max(np.nanmean(fms, axis=0)),
                                   "roc": np.max(np.nanmean(roc, axis=0)),
                                   "precision": np.max(np.nanmean(precision, axis=0)),
                                   "recall": np.max(np.nanmean(recall, axis=0)),
                                   "specificity": np.max(np.nanmean(specificity, axis=0)),
                                   }
        self.preds[method] = pred
        self.pred_probs[method] = pred_prob

    # def save_feature_importance(self, x, y, clf, feature_names):
    #     if clf is None:
    #         x_fs, feature_names = self.do_feature_selection_all(x.values, y, feature_names)
    #         model = ModelsHandler.get_model(clf)
    #         model.fit(x_fs, y)
    #         X = x_fs
    #     feature_scores = get_feature_scores(model_name, model, feature_names, x)
    #     return feature_scores

    def train(self, splits: list, clf: str, x_columns: list, feature_set: str, feature_importance: bool):
        self.splits = splits
        self.clf = clf
        self.method = 'default'

        acc = []
        fms = []
        roc = []
        precision = []
        recall = []
        specificity = []

        pred = {}
        pred_prob = {}
        feature_scores_fold = []
        k_range = None

        print("Model %s" % self.clf)
        print("=========================")
        for idx, fold in enumerate(self.splits):
            print("Processing fold: %i" % idx)
            x_train, y_train = fold['x_train'], fold['y_train'].ravel()
            x_test, y_test = fold['x_test'], fold['y_test'].ravel()
            labels_train, labels_test = fold['train_labels'], fold['test_labels']

            acc_scores = []
            fms_scores = []
            roc_scores = []
            p_scores = []  # precision
            r_scores = []  # recall
            spec_scores = []

            # getting feature selected x_train, x_test and the list of selected features
            x_train_fs, x_test_fs, selected_feature_names, k_range = \
                FeatureSelector().select_features(fold_data=fold, x_columns=x_columns, k_range=k_range)

            # fit the model
            model = ModelsHandler.get_model(clf)
            model = model.fit(x_train_fs, y_train)

            # make predictions
            yhat = model.predict(x_test_fs)
            yhat_probs = model.predict_proba(x_test_fs)
            for i in range(labels_test.shape[0]):
                pred[labels_test[i]] = yhat[i]
                pred_prob[labels_test[i]] = yhat_probs[i]

            # calculating metrics for each fold
            acc_scores, fms_scores, roc_scores, p_scores, r_scores, spec_scores = \
                self.compute_save_results(y_true=y_test, y_pred=yhat,
                                          y_prob=yhat_probs[:, 1], acc_saved=acc_scores,
                                          fms_saved=fms_scores, roc_saved=roc_scores,
                                          precision_saved=p_scores, recall_saved=r_scores, spec_saved=spec_scores)

            # adding every fold metric to the bigger list of metrics
            acc.append(acc_scores)
            fms.append(fms_scores)
            roc.append(roc_scores)
            precision.append(p_scores)
            recall.append(r_scores)
            specificity.append(spec_scores)

            # if feature_importance:
            #     feature_scores_fold.append(self.save_feature_importance(x=x_train_fs, y=None, clf=model, feature_names=selected_feature_names))

        self.save_results(method=self.method, acc=acc, fms=fms, roc=roc,
                          precision=precision, recall=recall, specificity=specificity,
                          pred=pred, pred_prob=pred_prob, k_range=k_range)

        self.feature_scores_fold[self.method] = feature_scores_fold

        # if feature_importance:  # get feature importance from the whole data
        #     self.feature_scores_all[self.method] = \
        #         self.save_feature_importance(X=self.X, y=self.y,
        #                                      model_name=model, model=None, feature_names=feature_names)

        return self

