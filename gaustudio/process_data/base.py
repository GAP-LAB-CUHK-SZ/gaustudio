from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """Abstract base class for datasets"""
    
    @abstractmethod
    def __init__(self):
        pass
        
    @abstractmethod
    def __len__(self):
        """Return dataset size"""
        pass
    
    @abstractmethod
    def __getitem__(self, index):
        """Return dataset element for given index"""
        pass
    
class BaseConverter(ABC):
    def __init__(self, dataset, target_dir):
        self.dataset = dataset
        self.target_dir = target_dir

    @abstractmethod
    def preprocess(self):
        """Cache required data from dataset""" 
        pass

    @abstractmethod
    def process(self):
        """Generate model"""
        pass

    @abstractmethod
    def postprocess(self, model):
        """Refine model""" 
        pass