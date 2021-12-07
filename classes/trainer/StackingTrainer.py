from classes.trainer.Trainer import Trainer
from classes.handlers.ModelsHandler import ModelsHandler
from classes.factories.DataSplitterFactory import DataSplitterFactory

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class StackingTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True):
        """
        manual stacking with cross-validation
        """
        self.method = 'ensemble'

        clfs = list(data.keys())
        some_clf = clfs[0]

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

        k_range = list(data[some_clf].best_k.values())[0]['k_range']

        splitter = DataSplitterFactory().get(self.aggregation_method)
        self.splits, self.x, self.y = splitter.make_splits(data=data, seed=self.seed)

        for idx, fold in enumerate(tqdm(self.splits, desc='Stacking Trainer')):
            # print("Processing fold: %i" % idx)
            x_train, y_train = fold['x_train'], fold['y_train'].ravel()
            x_test, y_test = fold['x_test'], fold['y_test'].ravel()
            labels_train, labels_test = fold['train_labels'], fold['test_labels']

            acc_scores = []
            fms_scores = []
            roc_scores = []
            p_scores = []  # precision
            r_scores = []  # recall
            spec_scores = []

            # train
            meta_model = ModelsHandler().get_model(clf)
            meta_model = meta_model.fit(x_train, y_train)

            # make predictions
            yhat_preds = meta_model.predict(x_test)
            yhat_pred_probs = meta_model.predict_proba(x_test)

            for i in range(labels_test.shape[0]):
                pred[labels_test[i]] = yhat_preds[i]
                pred_prob[labels_test[i]] = yhat_pred_probs[i]

            acc_scores, fms_scores, roc_scores, p_scores, r_scores, spec_scores = \
                self.compute_save_results(y_true=y_test, y_pred=yhat_preds,
                                          y_prob=yhat_pred_probs[:, 1], acc_saved=acc_scores,
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
