from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.base_preprocessor import autoPyTorchPreprocessingAlgorithm


class BaseEncoder(autoPyTorchPreprocessingAlgorithm):
    """
    Base class for encoder
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        if self.preprocessor is None:
            raise NotImplementedError()
        try:
            X = self.preprocessor.transform(X)
        except ValueError as msg:
            raise ValueError('{} in {}'.format(msg, self.__class__))
        return X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        return ConfigurationSpace()

