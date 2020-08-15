from typing import Any, Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.scaling import BaseScaler


class NoneScaler(BaseScaler):
    '''
    No scaling performed
    '''
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(NoneScaler, self).__init__()
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> BaseScaler:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
