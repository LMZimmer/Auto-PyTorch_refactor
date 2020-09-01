from typing import Any, Dict, Optional, Union

import numpy as np

import torch

from autoPyTorch.pipeline.components.preprocessing.scaling.base_scaler import BaseScaler


class NoScaler(BaseScaler):
    """
    No scaling performed
    """

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

        self.check_requirements(X, y)

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
        X.update({'scaler': self})
        return X

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'NoScaler',
            'name': 'No Scaler',
        }
