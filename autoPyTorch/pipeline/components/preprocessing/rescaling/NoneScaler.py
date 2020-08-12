from typing import Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.rescaling import BaseScaler


class NoneScaler(BaseScaler):
    '''
    No scaling performed
    '''
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(NoneScaler, self).__init__()
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NoneScaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
