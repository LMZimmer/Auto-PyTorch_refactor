from typing import Optional

import numpy as np

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
        raise NotImplementedError()

    def get_scheduler(self) -> _LRScheduler:
        """Return the underlying scheduler object.
        Returns:
            scheduler : the underlying scheduler object
        """
        assert self.scheduler is not None, "No scheduler was fit"
        return self.scheduler
