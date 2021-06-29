from classes.CrossValidator import CrossValidator
from classes.handlers.DataHandler import DataHandler
from classes.handlers.ModelsHandler import ModelsHandler
from classes.handlers.ParamsHandler import ParamsHandler


def main():
    # load_parameters to take the name of settings file (without .yaml extension)
    params = ParamsHandler.load_parameters('settings')
    mode = params["mode"]
    tasks = params["tasks"]
    classifiers = params["classifiers"]
    output_folder = params["output_folder"]  # how do I send this to PID extraction?

    # PID extraction option 1: call get_list_of_pids here with the PID_extraction_method, save pid list somewhere, then load_data does its work
    # this probably won't work since we have to specify the modalities and task for each task, which is handled inside DataHandler
    # extraction_method = params["PID_extraction_method"]
    # DataHandler.get_list_of_pids(mode=mode, modalities={"idk":"something?"}, task="choose a task from tasks", extraction_method=extraction_method)

    # PID extraction option 2: send extraction method through load_data
    extraction_method = params["PID_extraction_method"]
    tasks_data = DataHandler(mode=mode, extraction_method=extraction_method, output_folder=output_folder).load_data(tasks=tasks)
    models = ModelsHandler.get_models(classifiers)

    results = []
    for seed in range(params["seeds"]):
        print("\nProcessing seed {}\n".format(seed))

        """
        Single tasks
        * Each classifier process data stemming from each individual tasks
        * The output is a prediction for each task and classifier
        """
        if mode == "single_tasks":
            for task_data in tasks_data:
                for model in models:
                    metrics = CrossValidator.cross_validate(model, task_data)
                    results.append(metrics)

        """
        Task fusion
        * Each classifier process data stemming from all tasks at the same time
        * Individual classifiers are built for each modality using the same type of classifier
        * The individual task predictions are merged via averaging/stacking
        * The output is a prediction for each classifier
        """
        if mode == "fusion":
            for model in models:
                metrics = CrossValidator.cross_validate(model, tasks_data)
                results.append(metrics)

        """
        Models ensemble
        * For each task, all classifiers make a prediction on task data.
        * The individual classifiers predictions are merged via averaging/stacking
        * The output is a prediction for each task
        """
        if mode == "ensemble":
            for task_data in tasks_data:
                metrics = CrossValidator.cross_validate(models, task_data)
                results.append(metrics)


if __name__ == '__main__':
    main()
