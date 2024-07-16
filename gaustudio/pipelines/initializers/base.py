import os
from abc import ABC, abstractmethod

class BaseInitializer(ABC):
    """
    Abstract base class for initializers.
    """

    def __init__(self, initializer_config):
        """
        Initialize the BaseInitializer object.

        Args:
            workspace_dir (str): The path to the target directory for storing results.
        """
        self.initializer_config = initializer_config
        self.ws_dir = self.initializer_config['workspace_dir']
        os.makedirs(self.ws_dir, exist_ok=True)

    def setup(self):
        pass
    
    def cache_dataset(self, dataset=None):
        """
        Cache required data from the dataset.

        Args:
            dataset: The dataset from which data needs to be cached.
        """
        pass

    def process_dataset(self):
        """
        Process the dataset to generate an initial model.
        """
        pass

    @abstractmethod
    def build_model(self, model):
        """
        Refine the initial model.
        """
        pass

    def __call__(self, model, dataset=None, overwrite=False):
        """
        Entry point to run the initialization and model refinement process.

        Args:
            model: The initial model object to be initialized and refined.
            dataset: The dataset object to be used for initialization.
            overwrite (bool): Flag to indicate if existing results should be overwritten.

        Returns:
            The refined model object.
        """
        if overwrite or not self.should_skip():
            self.cache_dataset(dataset)
            self.process_dataset()
        model = self.build_model(model)
        return model

    def should_skip(self):
        """
        Determine if the initialization process should be skipped based on 
        the existence of prior results and the overwrite flag.

        Returns:
            bool: Whether to skip the initialization process or not.
        """
        return False