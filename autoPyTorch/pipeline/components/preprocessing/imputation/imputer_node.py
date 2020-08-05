from ConfigSpace.configuration_space import ConfigurationSpace
from autoPyTorch.pipeline.base_component import AutoPytorchComponent
from sklearn.impute import SimpleImputer
import numpy as np
from typing import Optional

TBaseImputer = TypeVar("TBaseImputer", bound="BaseImputer")


class BaseImputer(AutoPytorchComponent):
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseImputer":
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X).astype(int)
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
    
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[dict] = None) -> ConfigurationSpace:
        return ConfigurationSpace()