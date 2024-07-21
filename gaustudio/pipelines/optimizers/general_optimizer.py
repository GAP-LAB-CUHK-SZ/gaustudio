from abc import ABC, abstractmethod
import torch
import torch_scatter
import time
from torch import nn
from gaustudio.pipelines.optimizers.base import BaseOptimizer
from gaustudio.pipelines import optimizers

@optimizers.register('general')
class GeneralOptimizer(BaseOptimizer):
    def _initialize_internal_state(self):
        """Initialize any optimizer-specific internal state."""
        if 'params' in self.config:
            self.param_groups = []
            for name, args in self.config["params"].items():
                parameter = nn.Parameter(getattr(self.model, '_'+name))
                parameter.requires_grad = True
                setattr(self.model, '_'+name, parameter)
                self.param_groups.append({'params': getattr(self.model, '_'+name), 'name': name, **args})
        else:
            self.param_groups = self.model.parameters()