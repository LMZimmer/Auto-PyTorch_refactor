from typing import Any, Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.encoding import BaseEncoder


class NoneEncoder(BaseEncoder):
    """
    Don't perform encoding on categorical features

    Parameters

    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(NoneEncoder, self).__init__()
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> BaseEncoder:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
