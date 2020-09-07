from typing import Any, Dict, Union

import numpy as np

import torch

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


class BaseScaler(autoPyTorchPreprocessingComponent):
    """
    Provides abstract class interface for Scalers in AutoPytorch
    """

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted preprocessor into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.column_transformer is None:
            raise ValueError("cant call transform on {} without fitting first."
                             .format(self.__class__.__name__))
        X.update({'scaler': self})
        return X

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
        if self.column_transformer is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))
        X = self.column_transformer.transform(X)
        return X

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
        if 'numerical_columns' not in X:
            raise ValueError("To fit a scaler, the fit dictionary "
                             "must contain a list of the numerical "
                             "columns of the data but only contains {}".format(X.keys())
                             )
