from typing import List

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


class ModelsHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_models(classifiers: List) -> List:
        models = []

        for model in classifiers:
            if model == 'RandomForest':
                models.append(RandomForestClassifier())
            elif model == 'GausNaiveBayes':
                models.append(GaussianNB())
            elif model == 'LogReg':
                models.append(LogisticRegression(max_iter=80000))
            elif model == 'dummy':
                models.append(DummyClassifier())
            else:
                raise ("invalid classifier: %s", model)
        return models

    @staticmethod
    def get_model(classifier: str) -> object:
        if classifier == 'RandomForest':
            return RandomForestClassifier()
        elif classifier == 'GausNaiveBayes':
            return GaussianNB()
        elif classifier == 'LogReg':
            return LogisticRegression(max_iter=80000)
        elif classifier == 'dummy':
            return DummyClassifier()
        else:
            raise ("invalid classifier: %s", classifier)
