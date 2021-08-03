from classes.handlers.ParamsHandler import ParamsHandler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
'''
Abstract class Trainer
'''


class Trainer:
    def __init__(self):
        self.clf = None
        self.splits = None
        self.preds = {}
        self.pred_probs = {}
        self.results = {}
        self.method = 'default'
        self.k_range = None
        self.best_k = {}
        self.best_score = {}
        self.feature_scores_fold = {}
        self.feature_set = None


    def train(self, folds: list, model: object, x_columns: list, feature_set: str):
        pass

    def calculate_task_fusion_results(self, data, model):
        pass
