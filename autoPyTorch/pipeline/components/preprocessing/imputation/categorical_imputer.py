from ConfigSpace.configuration_space import ConfigurationSpace
from autoPyTorch.pipeline.base_component import AutoPytorchComponent
from sklearn.impute import SimpleImputer
import numpy as np
from typing import Optional


class CategoricalImputer(AutoPytorchComponent):
    '''
    Impute missing values for categorical columns with {FILL_VALUE}
    '''
    FILL_VALUE = 2

    def __init__(self, random_state: int = None):
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> CategoricalImputer:
        self.preprocessor = SimpleImputer(strategy='constant', fill_value=FILL_VALUE, copy=False)
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
