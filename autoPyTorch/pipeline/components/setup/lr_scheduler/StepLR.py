from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class StepLR(autoPyTorchSetupComponent):
    """
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    Notice that such decay can happen simultaneously with other changes to the learning
    rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        step_size (int) – Period of learning rate decay.
        gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.

    """
    def __init__(
        self,
        step_size: int,
        gamma: float,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.gamma = gamma
        self.step_size = step_size
        self.random_state = random_state
        self.scheduler = None  # type: Optional[_LRScheduler]

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any
            ) -> autoPyTorchSetupComponent:
        """
        Sets the scheduler component choice as CosineAnnealingWarmRestarts

        Args:
            X (np.ndarray): input features
            y (npndarray): target features

        Returns:
            A instance of self
        """
        import torch.optim.lr_scheduler

        # Make sure there is an optimizer
        if 'optimizer' not in fit_params:
            raise ValueError('Cannot use scheduler without an optimizer to wrap')

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=fit_params['optimizer'],
            step_size=int(self.step_size),
            gamma=float(self.gamma),
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
            "gamma", 0.001, 0.9, default_value=0.1)
        step_size = UniformIntegerHyperparameter(
            "step_size", 1, 10, default_value=5)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([gamma, step_size])
        return cs
