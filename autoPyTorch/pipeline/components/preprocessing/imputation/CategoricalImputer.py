from typing import Optional, Union

import numpy as np

from sklearn.impute import SimpleImputer

from autoPyTorch.pipeline.components.preprocessing.imputation.base import BaseImputer


class CategoricalImputer(BaseImputer):
    '''
    Impute missing values for categorical columns with '!missing!'
    '''
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.preprocessor = SimpleImputer(strategy='constant', fill_value='!missing!', missing_values='nan', copy=False)
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseImputer":
        self.preprocessor.fit(X.astype(object))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self.preprocessor.transform(X.astype(object))
        return X
