import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from typing import Optional, TypeVar

TBaseScaler = TypeVar("TBaseScaler", bound="BaseScaler")

from autoPyTorch.pipeline.base_component import AutoPytorchComponent

class BaseScaler(AutoPytorchComponent):
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TBaseScaler:
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
    def get_hyperparameter_search_space(dataset_properties: Optional[dict] = None) -> CS.ConfigurationSpace:
        return ConfigurationSpace()