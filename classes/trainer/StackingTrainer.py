from classes.trainer.Trainer import Trainer
from classes.handlers.ModelsHandler import ModelsHandler

import numpy as np


class StackingTrainer(Trainer):
    def __init__(self):
        super().__init__()


    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True):
        """
        manual stacking with cross-validation
        """

        clfs = list(data.keys())
        some_clf = clfs[0]
        n_folds = len(data[some_clf].fold_preds_train)

        # defining metrics
        acc = []
        fms = []
        roc = []
        precision = []
        recall = []
        specificity = []

        pred = {}
        pred_prob = {}
        feature_scores_fold = []
        k_range = range(len(clfs))  # placeholder value for k-range so that it does something

        for idx in range(n_folds):
            acc_scores = []
            fms_scores = []
            roc_scores = []
            p_scores = []  # precision
            r_scores = []  # recall
            spec_scores = []

            # training data extraction
            pids_train = list(data[some_clf].fold_preds_train[idx].keys())

            train_preds_fold = {pid: [data[clf].fold_preds_train[idx][pid] for clf in clfs] for pid in pids_train}
            train_x_preds_fold = np.array(list(train_preds_fold.values()))
            train_y_preds_fold = data[some_clf].splits[idx]['y_train']

            # test data extraction
            pids_test = list(data[some_clf].fold_preds_test[idx].keys())
            test_labels = data[some_clf].splits[idx]['test_labels']

            test_preds_fold = {pid: [data[clf].fold_preds_test[idx][pid] for clf in clfs] for pid in pids_test}
            test_x_preds_fold = np.array(list(test_preds_fold.values()))
            test_y_preds_fold = data[some_clf].splits[idx]['y_test']


            # fit the meta classifier
            meta_model = ModelsHandler().get_model(clf)
            meta_model = meta_model.fit(train_x_preds_fold, train_y_preds_fold)

            # meta_clf.score(test_x_preds_fold, test_y_preds_fold)
            yhat_preds = meta_model.predict(test_x_preds_fold)
            yhat_preds_probs = meta_model.predict_proba(test_x_preds_fold)

            for i in range(test_labels.shape[0]):
                pred[test_labels[i]] = yhat_preds[i]
                pred_prob[test_labels[i]] = yhat_preds_probs[i]


            # calculating metrics for each fold
            acc_scores, fms_scores, roc_scores, p_scores, r_scores, spec_scores = \
                self.compute_save_results(y_true=test_y_preds_fold, y_pred=yhat_preds,
                                          y_prob=yhat_preds_probs[:, 1], acc_saved=acc_scores,
                                          fms_saved=fms_scores, roc_saved=roc_scores,
                                          precision_saved=p_scores, recall_saved=r_scores, spec_saved=spec_scores)

            # adding every fold metric to the bigger list of metrics
            acc.append(acc_scores)
            fms.append(fms_scores)
            roc.append(roc_scores)
            precision.append(p_scores)
            recall.append(r_scores)
            specificity.append(spec_scores)

            '''
            # if feature_importance:
            #     feature_scores_fold.append(self.save_feature_importance(x=x_train_fs, y=None, clf=model,
            #                                                             feature_names=selected_feature_names))
            '''



        self.save_results(method=self.method, acc=acc, fms=fms, roc=roc,
                          precision=precision, recall=recall, specificity=specificity,
                          pred=pred, pred_prob=pred_prob, k_range=k_range)

        self.feature_scores_fold[self.method] = feature_scores_fold

        '''
        # if feature_importance:  # get feature importance from the whole data
        #     self.feature_scores_all[self.method] = \
        #         self.save_feature_importance(x=self.x, y=self.y,
        #                                      clf=clf, feature_names=feature_names)
        '''

        return self
