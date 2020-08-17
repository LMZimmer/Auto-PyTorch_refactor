from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.encoding import BaseEncoder


class NoneEncoder(BaseEncoder):
    """
    Don't perform encoding on categorical features

    Parameters

    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(NoneEncoder, self).__init__()
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> BaseEncoder:
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

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'NoneEncoder',
            'name': 'None Encoder',
        }
