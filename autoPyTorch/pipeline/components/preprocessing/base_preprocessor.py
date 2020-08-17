from typing import Any, Optional

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchPreprocessingAlgorithm(TransformerMixin, autoPyTorchComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    def __init__(self) -> None:
        self.preprocessor: BaseEstimator = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> autoPyTorchComponent:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        raise NotImplementedError

    def get_preprocessor(self) -> BaseEstimator:
        return self.preprocessor
