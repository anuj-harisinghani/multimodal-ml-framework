from classes.cv.CrossValidator import CrossValidator
from classes.handlers.DataHandler import DataHandler
from classes.handlers.ParamsHandler import ParamsHandler

import os
import warnings
warnings.filterwarnings("ignore")


def main():
    # load_parameters to take the name of settings file (without .yaml extension)
    params = ParamsHandler.load_parameters('settings')
    seeds = params["seeds"]
    mode = params["mode"]
    tasks = params["tasks"]
    classifiers = params["classifiers"]
    output_folder = params["output_folder"]
    extraction_method = params["PID_extraction_method"]

    path = os.path.join(os.getcwd(), 'results', output_folder)
    if not os.path.exists(path):
        os.mkdir(path)

    # getting the data from DataHandler and models from ModelsHandler
    tasks_data = DataHandler(mode=mode, output_folder=output_folder, extraction_method=extraction_method).load_data(tasks=tasks)

    for seed in range(seeds):
        # running CrossValidator on the extracted data
        print("\nSeed: %s" % str(seed))
        cv = CrossValidator(mode, seed, classifiers)
        cv.cross_validate(tasks_data=tasks_data)


if __name__ == '__main__':
    main()
