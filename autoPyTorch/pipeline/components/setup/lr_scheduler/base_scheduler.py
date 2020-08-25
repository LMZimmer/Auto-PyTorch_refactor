from typing import Any, Dict, Optional

import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseLRComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for schedulers
    in Auto-Pytorch"""

    def __init__(self) -> None:
        self.scheduler = None  # type: Optional[_LRScheduler]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'lr_scheduler': self.scheduler})
        return X

    def get_scheduler(self) -> _LRScheduler:
        """Return the underlying scheduler object.
        Returns:
            scheduler : the underlying scheduler object
        """
        assert self.scheduler is not None, "No scheduler was fit"
        return self.scheduler

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """

        # make sure the parent requirements are honored
        super().check_requirements(X, y)

        # The fit dictionary must have an optimizer, that the LR will wrap
        if 'optimizer' not in X or not isinstance(X['optimizer'], Optimizer):
            raise ValueError("To fit a learning rate scheduler, the fit dictionary "
                             "Must contain a valid optimizer that inherits from "
                             "torch.optim.Optimizer, yet X only contains {}.".format(
                                 X
                             )
                             )

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.scheduler.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('scheduler', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
