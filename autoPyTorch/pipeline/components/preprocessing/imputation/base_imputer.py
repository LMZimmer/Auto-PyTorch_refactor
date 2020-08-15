from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.base_preprocessor import autoPyTorchPreprocessingAlgorithm


class BaseImputer(autoPyTorchPreprocessingAlgorithm):
    '''
    Provides abstract class interface for Imputers in AutoPyTorch
    '''
    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            **fit_params: Any) -> autoPyTorchPreprocessingAlgorithm:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """
        self.preprocessor.fit(X)
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
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        return ConfigurationSpace()
