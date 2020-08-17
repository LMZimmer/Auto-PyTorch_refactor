from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.base_preprocessor import autoPyTorchPreprocessingAlgorithm


class BaseImputer(autoPyTorchPreprocessingAlgorithm):
    """
    Provides abstract class interface for Imputers in AutoPyTorch
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
        X = self.preprocessor.transform(X)
        return X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        return ConfigurationSpace()
