from classes.handlers.ParamsHandler import ParamsHandler

"""
Abstract class Trainer
"""


class Trainer:
    def __init__(self):
        params = ParamsHandler.load_parameters('settings')
        self.mode = params['mode']
        self.clf = None
        self.splits = None
        self.data = None
        self.x = None
        self.y = None
        self.labels = None
        self.feature_set = None
        self.seed = None

        self.preds = {}
        self.pred_probs = {}
        self.results = {}
        self.method = None
        self.k_range = None
        self.best_k = {}
        self.best_score = {}
        self.feature_scores_fold = {}
        self.feature_scores_all = {}


    def train(self, data: dict, clf: str, feature_set: str, feature_importance: bool):
        pass

    def calculate_task_fusion_results(self, data, seed):
        pass
