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
            if model == 'rf':
                models.append(RandomForestClassifier())
            elif model == 'gnb':
                models.append(GaussianNB())
            elif model == 'lr':
                models.append(LogisticRegression())
            elif model == 'dummy':
                models.append(DummyClassifier())
            else:
                raise("invalid classifier: %s", model)
        return models
