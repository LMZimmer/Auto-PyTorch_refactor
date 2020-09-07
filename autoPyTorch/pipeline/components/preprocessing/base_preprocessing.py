from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

import torch

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchPreprocessingComponent(autoPyTorchComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    def __init__(self) -> None:
        self.preprocessor: Union[Dict[str, BaseEstimator], Optional[BaseEstimator]] = None
        self.column_transformer: Optional[ColumnTransformer] = None

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
        super().check_requirements(X, y)
        if 'train' not in X:
            raise ValueError("To fit a preprocessor, the fit dictionary "
                             "Must contain a reference to the training data"
                             )

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]): Describes the dataset
               to work on

        Returns:
            ConfigurationSpace: The configuration space of this algorithm.
        """
        return ConfigurationSpace()
