from classes.cv.CrossValidator import CrossValidator

'''
Abstract class Trainer
'''


class Trainer:
    def __init__(self, x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels


    def train(self, folds: list):
        pass
