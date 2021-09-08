from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


class ModelsHandler:
    def __init__(self):
        # empty init
        pass

    @staticmethod
    def get_models(classifiers: list) -> list:
        models = []

        for model in classifiers:
            if model == 'RandomForest':
                models.append(RandomForestClassifier())
            elif model == 'GausNaiveBayes':
                models.append(GaussianNB())
            elif model == 'LogReg':
                models.append(LogisticRegression())
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
            return LogisticRegression()
        elif classifier == 'dummy':
            return DummyClassifier()
        else:
            raise ("invalid classifier: %s", classifier)
