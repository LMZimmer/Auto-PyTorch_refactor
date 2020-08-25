from typing import Any, Dict, Optional, Union

import numpy as np

import torch
from autoPyTorch.pipeline.components.preprocessing.scaling import BaseScaler


class NoneScaler(BaseScaler):
    """
    No scaling performed
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(NoneScaler, self).__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:
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

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        return X

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> None:
        raise NotImplementedError

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'NoneScaler',
            'name': 'None Scaler',
        }
