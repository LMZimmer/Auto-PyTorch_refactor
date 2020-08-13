import numpy as np

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchSetupComponent(autoPyTorchComponent):
    """Provide an abstract interface for schedulers
    in Auto-Pytorch"""

    def __init__(self) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        raise NotImplementedError()
