from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent


class CosineAnnealingLR(BaseLRComponent):
    """
    Set the learning rate of each parameter group using a cosine annealing schedule

    Args:
        T_max (int): Maximum number of iterations.

    """
    def __init__(
        self,
        T_max: int,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.T_max = T_max
        self.random_state = random_state
        self.scheduler = None  # type: Optional[_LRScheduler]

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any
            ) -> BaseLRComponent:
        """
        Sets the scheduler component choice as CosineAnnealingWarmRestarts

        Args:
            X (np.ndarray): input features
            y (npndarray): target features

        Returns:
            A instance of self
        """

        # Make sure there is an optimizer
        if 'optimizer' not in fit_params:
            raise ValueError('Cannot use scheduler without an optimizer to wrap')

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=fit_params['optimizer'],
            T_max=int(self.T_max)
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'CosineAnnealingWarmRestarts',
            'name': 'Cosine Annealing WarmRestarts',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None
                                        ) -> ConfigurationSpace:
        T_max = UniformIntegerHyperparameter(
            "T_max", 10, 500, default_value=200)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([T_max])
        return cs
