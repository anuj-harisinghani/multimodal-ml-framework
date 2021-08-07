from classes.handlers.ParamsHandler import ParamsHandler


class DataSplitter:
    def __init__(self):
        params = ParamsHandler.load_parameters('settings')
        self.random_seed = params['random_seed']
        self.mode = params['mode']
        self.nfolds = params['folds']

    def make_splits(self, data: dict) -> list:
        pass
