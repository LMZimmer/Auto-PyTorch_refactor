from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

import torch

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchPreprocessingComponent(autoPyTorchComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None) -> None:
        self.random_state = random_state
        self.preprocessor: Union[Dict[str, BaseEstimator], Optional[BaseEstimator]] = None
        self.column_transformer: Optional[ColumnTransformer] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchComponent:
        """
        The fit function calls the fit function of the underlying model
        and returns the self.
        Args:
            X (Dict[str, Any]): 'X' dictionary
            y (Any): should be none

        Returns:
            instance of self
        """
        raise NotImplementedError()

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted preprocessor into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        raise NotImplementedError()

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        """
        Makes the autoPyTorchPreprocessingComponent Callable. Calling the component
        calls the transform function of the underlying preprocessor and
        returns the transformed array.
        Args:
            X (Union[np.ndarray, torch.tensor]): input data tensor

        Returns:
            Union[np.ndarray, torch.tensor]: Transformed data tensor
        """
        raise NotImplementedError()

    def get_column_transformer(self) -> ColumnTransformer:
        """
        Get fitted column transformer that is wrapped around
        the sklearn preprocessor. Can only be called if fit()
        has been called on the object.
        Returns:
            BaseEstimator: Fitted sklearn column transformer
        """
        return self.column_transformer
