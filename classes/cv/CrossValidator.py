from abc import ABC
from typing import Dict, Union, List

import pandas as pd

from classes.models.Model import Model
from classes.tasks.Task import Task
from classes.cv.DataSplitter import DataSplitter


class CrossValidator(ABC):
    def __init__(self):
        pass

    @staticmethod
    # def cross_validate(self, m: Union[Model, List], t: Union[Task, List]) -> Dict:
    #     pass
    def cross_validate(mode: str, tasks_data: dict) -> dict:
        for task in tasks_data.keys():
            for modality, modality_data in tasks_data[task].items():
                splits = DataSplitter.make_splits(modality_data, modality_data['labels'])

        return {}