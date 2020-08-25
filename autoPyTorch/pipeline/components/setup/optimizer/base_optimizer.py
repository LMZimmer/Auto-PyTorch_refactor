from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseOptimizerComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for Pytorch Optimizers
    in Auto-Pytorch"""

    def __init__(self) -> None:
        self.optimizer = None  # type: Optional[Optimizer]

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'optimizer': self.optimizer})
        return X

    def get_optimizer(self) -> Optimizer:
        """Return the underlying Optimizer object.
        Returns:
            model : the underlying Optimizer object
        """
        assert self.optimizer is not None, "No optimizer was fitted"
        return self.optimizer

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """ This common utility makes sure that the input dictionary X,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X, y)

        # For the optimizer, we need the network to wrap
        if 'network' not in X or not isinstance(X['network'], torch.nn.Module):
            raise ValueError("Could not parse the network in the fit dictionary "
                             "To fit a optimizer, the network is needed to define "
                             "which parameters to wrap, yet the dict contains only: {}".format(
                                 X
                             )
                             )

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.optimizer.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('optimizer', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
