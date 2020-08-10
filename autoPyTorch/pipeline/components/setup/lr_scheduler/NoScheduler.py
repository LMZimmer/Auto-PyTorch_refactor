from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class NoScheduler(autoPyTorchSetupComponent):
    """
    Performs no scheduling via a LambdaLR with lambda==1.

    """
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.lr_lambda = lambda epoch: 1
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

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=fit_params['optimizer'],
            lr_lambda=self.lr_lambda,
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
        cs = ConfigurationSpace()
        return cs
