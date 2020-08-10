import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from sklearn.preprocessing import Normalizer as SklearnNormalizer, MinMaxScaler as SklearnMinMaxScaler, StandardScaler as SklearnStandardScaler
from typing import Optional, Tuple, Union

from autoPyTorch.pipeline.components.preprocessing.rescaling import BaseScaler


class Normalizer(BaseScaler):
    '''
    Normalise columns/features using {l1, l2, max}
    '''
    def __init__(self, random_state: Optional[int] = None, norm: str = 'l2'):
        self.random_state = random_state
        self.preprocessor = SklearnNormalizer(norm=norm, copy=False)
    
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[dict] = None) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        norm = CSH.CategoricalHyperparameter("norm", ["l1", "l2", "max"], default_value="l2")
        cs.add_hyperparameter(strategy)
        return cs


class MinMaxScaler(BaseScaler):
    '''
    Scale numerical columns/features into feature_range
    '''
    def __init__(self, random_state: Optional[int] = None, feature_range: Tuple[Union[int, float], Union[int, float]] = (0,1)):
        self.random_state = random_state
        self.preprocessor = SklearnMinMaxScaler(feature_range=feature_range, copy=False)


class StandardScaler(BaseScaler):
    '''
    Standardise numerical columns/features by removing mean and scaling to unit/variance
    '''
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.preprocessor = SklearnStandardScaler(with_mean=True, with_std=True, copy=False)
    
    
class NoneScaler(BaseScaler):
    '''
    No scaling performed
    '''
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NoneScaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
    
