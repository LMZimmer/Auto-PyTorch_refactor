from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np
import torch

from autoPyTorch.pipeline.components.preprocessing.base_preprocessor import autoPyTorchPreprocessingComponent


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
        if self.preprocessor is None:
            raise ValueError("cant call transform on {} without fitting first.".format(self.__class__.__name__))
        X.update({'scaler': self.preprocessor})
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
        if self.preprocessor is None:
            raise ValueError("cant call {} without fitting the preprocessor first.".format(self.__class__.__name__))
        X = self.preprocessor.transform(X)
        return X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        return ConfigurationSpace()
