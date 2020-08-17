from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.base_preprocessor import autoPyTorchPreprocessingAlgorithm


class BaseScaler(autoPyTorchPreprocessingAlgorithm):
    """
    Provides abstract class interface for Scalers in AutoPytorch
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
