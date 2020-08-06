from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.impute import SimpleImputer
import numpy as np
from typing import Optional

from autoPyTorch.pipeline.components.preprocessing.base_preprocessor import autoPyTorchPreprocessingAlgorithm


class BaseImputer(autoPyTorchPreprocessingAlgorithm):
    '''
    Base 
    '''
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseImputer":
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
    
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[dict] = None) -> ConfigurationSpace:
        return ConfigurationSpace()