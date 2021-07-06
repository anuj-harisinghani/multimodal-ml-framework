class DataSplitterFactory:

    def __init__(self):
        self.__data_splitters = {
            "single_task": SingleTaskDataSplitter,
            "task_fusion": TaskFusionDataSplitter
        }

    def get(self, data_splitter_type: str) -> DataSplitter:
        if not data_splitter_type in self.__data_splitters.keys():
            raise ValueError("Data splitter '{}' not supported! Supported splitters are: {}"
                             .format(data_splitter_type, self.__data_splitters.keys()))
        return self.__data_splitters[data_splitter_type]()
