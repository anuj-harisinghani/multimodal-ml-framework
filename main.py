from classes.cv.CrossValidator import CrossValidator
from classes.handlers.DataHandler import DataHandler
from classes.handlers.ModelsHandler import ModelsHandler
from classes.handlers.ParamsHandler import ParamsHandler
from classes.handlers.PIDExtractor import PIDExtractor

import os


def main():
    # load_parameters to take the name of settings file (without .yaml extension)
    params = ParamsHandler.load_parameters('settings')
    mode = params["mode"]
    tasks = params["tasks"]
    classifiers = params["classifiers"]

    output_folder = params["output_folder"]
    extraction_method = params["PID_extraction_method"]

    # getting the data from DataHandler and models from ModelsHandler
    tasks_data = DataHandler(mode=mode, output_folder=output_folder, extraction_method=extraction_method).load_data(tasks=tasks)
    # models = ModelsHandler.get_models(classifiers)

    results = []
    cv = CrossValidator(mode, classifiers)
    results = cv.cross_validate(tasks_data=tasks_data)

    # for seed in range(params["seeds"]):
    #     print("\nProcessing seed {}\n".format(seed))
    #
    #     """
    #     Single tasks
    #     * Each classifier process data stemming from each individual tasks
    #     * The output is a prediction for each task and classifier
    #     """
    #     if mode == "single_tasks":
    #         for task_data in tasks_data:
    #             for model in models:
    #                 metrics = CrossValidator.cross_validate(model, task_data)
    #                 results.append(metrics)
    #
    #     """
    #     Task fusion
    #     * Each classifier process data stemming from all tasks at the same time
    #     * Individual classifiers are built for each modality using the same type of classifier
    #     * The individual task predictions are merged via averaging/stacking
    #     * The output is a prediction for each classifier
    #     """
    #     if mode == "fusion":
    #         for model in models:
    #             metrics = CrossValidator.cross_validate(model, tasks_data)
    #             results.append(metrics)
    #
    #     """
    #     Models ensemble
    #     * For each task, all classifiers make a prediction on task data.
    #     * The individual classifiers predictions are merged via averaging/stacking
    #     * The output is a prediction for each task
    #     """
    #     if mode == "ensemble":
    #         for task_data in tasks_data:
    #             metrics = CrossValidator.cross_validate(models, task_data)
    #             results.append(metrics)


if __name__ == '__main__':
    main()
