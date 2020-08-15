from typing import Optional, Tuple, Union

import numpy as np

from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from autoPyTorch.pipeline.components.preprocessing.scaling import BaseScaler


class MinMaxScaler(BaseScaler):
    '''
    Scale numerical columns/features into feature_range
    '''
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 feature_range: Tuple[Union[int, float], Union[int, float]] = (0, 1)):
        self.random_state = random_state
        self.preprocessor = SklearnMinMaxScaler(feature_range=feature_range, copy=False)
