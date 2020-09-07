from typing import Any, Callable, Dict, List, Optional

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import torch

from autoPyTorch.pipeline.components.setup.network_initializer.base_network_initializer import (
    BaseNetworkInitializerComponent
)


class OrthogonalInit(BaseNetworkInitializerComponent):
    """
    Fills the input Tensor with a (semi) orthogonal matrix
    """

    def weights_init(self) -> Callable:
        """Returns the actual PyTorch model, that is dynamically created
        from a self.config object.

        self.config is a dictionary created form a given config in the config space.
        It contains the necessary information to build a network.
        """
        def initialization(m):
            if isinstance(m, (torch.nn.Conv1d,
                              torch.nn.Conv2d,
                              torch.nn.Conv3d,
                              torch.nn.Linear)):
                torch.nn.init.orthogonal_(m.weight.data)
                if m.bias is not None and self.bias_strategy == 'Zero':
                    torch.nn.init.constant_(m.bias.data, 0.0)
        return initialization
