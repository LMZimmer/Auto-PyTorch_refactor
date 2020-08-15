from typing import Optional, Union

import numpy as np

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from autoPyTorch.pipeline.components.preprocessing.scaling import BaseScaler


class StandardScaler(BaseScaler):
    '''
    Standardise numerical columns/features by removing mean and scaling to unit/variance
    '''
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.random_state = random_state
        self.preprocessor = SklearnStandardScaler(with_mean=True, with_std=True, copy=False)
