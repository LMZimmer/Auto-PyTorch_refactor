from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler import BaseLRComponent


class NoScheduler(BaseLRComponent):
    """
    Performs no scheduling via a LambdaLR with lambda==1.

    """
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None
    ):

        super().__init__()
        self.random_state = random_state
        self.scheduler = None  # type: Optional[_LRScheduler]

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseLRComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Make sure there is an optimizer
        self.check_requirements(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'NoScheduler',
            'name': 'No LR Scheduling',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
