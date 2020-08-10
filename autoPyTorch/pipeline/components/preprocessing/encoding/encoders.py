from ConfigSpace.configuration_space import ConfigurationSpace
import numpy as np
from typing import Optional
from sklearn.preprocessing import OneHotEncoder as OHE, OrdinalEncoder as OE
from . import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """Encode categorical features as a one-hot numerical array"""
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.preprocessor = OHE(categories='auto', sparse=False, handle_unknown='ignore')


class OrdinalEncoder(BaseEncoder):
    """Encode categorical features as a one-hot numerical array"""
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.preprocessor = OE(categories='auto')

class NoneEncoder(BaseEncoder):
    """Don't perform encoding on categorical features"""
    def __init__(self, random_state: Optional[int] = None):
        super(NoneEncoder, self).__init__()
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NoneEncoder":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
 