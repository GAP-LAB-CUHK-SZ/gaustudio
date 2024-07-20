from abc import ABC, abstractmethod
import torch
import torch_scatter
import time
import torch.optim as optim

class BaseOptimizer(ABC):
    """An abstract base class for optimizers."""

    def __init__(self, 
            config,
            **kwargs):
        self.config = config
        self.model = config["model"]       
        self._initialize_internal_state()
        self.setup_optimizer()
        
    def setup_optimizer(self):
        self._optimizer = getattr(torch.optim, self.config["optimizer_name"])(self.param_groups, **self.config['args'])
    
    @abstractmethod
    def _initialize_internal_state(self):
        """Initialize any optimizer-specific internal state."""
        pass
            
    def step(self):
        """Performs a single optimization step."""
        self._optimizer.step()

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        self._optimizer.zero_grad()
