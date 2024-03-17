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
    
    @abstractmethod
    def preprocess(self):
        """
        Cache required data from the dataset.
        """
        pass

    @abstractmethod
    def process(self):
        """
        Generate an initial model.
        """
        pass

    @abstractmethod
    def postprocess(self, model):
        """
        Refine the initial model.
        """
        pass

    def __call__(self, pcd, dataset):
        pass