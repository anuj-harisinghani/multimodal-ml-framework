import yaml
import os


class ParamsHandler:
    def __init__(self):
        # not required
        pass

    @staticmethod
    def load_parameters(filename: str) -> dict:
        with open(os.path.join(os.getcwd(), 'params', filename + '.yaml')) as file:
            config = yaml.safe_load(file)

        return config

    @staticmethod
    def save_parameters(params: dict, path: str):
        with open((path + '.yaml'), 'w') as file:
            yaml.dump(params, file)

    @staticmethod
    def name_output_folder(filename: str):
        params = ParamsHandler.load_parameters(filename)

        mode = params["mode"]
        aggregation_method = params["aggregation_method"]
        meta_clf = params["meta_classifier"]
        dataset_name = params["dataset"]
        output_folder = params["output_folder"]

        # naming the output_folder based on the configuration chosen
        if output_folder == "":
            output_folder = "{}_".format(mode)

            if mode == 'data_ensemble' or mode == 'models_ensemble':
                output_folder += "{}_".format(aggregation_method)

                if aggregation_method == 'stack':
                    output_folder += "{}_".format(meta_clf)

            folder = os.path.join(os.getcwd(), 'results', dataset_name)
            previous_output_folders = [int(i.split('_')[-1]) for i in os.listdir(folder) if i.startswith(output_folder)]
            if not previous_output_folders:
                output_folder += str(1)
            else:
                output_folder += str(max(previous_output_folders) + 1)

        return output_folder
