from ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.pipeline.base_component import AutoPytorchComponent
from sklearn.impute import SimpleImputer
import numpy as np
from typing import Optional

FILL_VALUE = 2

class NumericalImputer(AutoPytorchComponent):
    '''
    Impute missing values for numerical columns using one of {'mean', 'median', 'most_frequent', 'constant_zero'} strategy.
    '''
    def __init__(self, random_state: int = None, strategy: str = 'mean'):
        self.random_state = random_state
        self.strategy = strategy
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> NumericalImputer:
        if self.strategy == 'constant_zero':
            self.preprocessor = SimpleImputer(strategy='constant', fill_value=FILL_VALUE, verbose=self.verbose, copy=False)
        else:
            self.preprocessor = SimpleImputer(strategy=self.strategy, verbose=self.verbose, copy=False)

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
        cs = CS.ConfigurationSpace()
        strategy = CSH.CategoricalHyperparameter("strategy", ["mean", "median", "most_frequent", "constant_zero"], default_value="mean")
        cs.add_hyperparameter(strategy)
        return cs
