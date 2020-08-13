from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent


class ExponentialLR(BaseLRComponent):
    """
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        gamma (float): Multiplicative factor of learning rate decay.

    """
    def __init__(
        self,
        gamma: float,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.gamma = gamma
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

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=fit_params['optimizer'],
            gamma=float(self.gamma)
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
        gamma = UniformFloatHyperparameter(
            "gamma", 0.7, 0.9999, default_value=0.9)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([gamma])
        return cs
