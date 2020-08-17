from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder as OHE

from autoPyTorch.pipeline.components.preprocessing.encoding import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """
    Encode categorical features as a one-hot numerical array
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.random_state = random_state
        self.preprocessor: Optional[BaseEstimator] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> BaseEncoder:
        self.preprocessor = OHE(categories='auto', sparse=False, handle_unknown='error')
        self.preprocessor.fit(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'OneHotEncoder',
            'name': 'One Hot Encoder',
        }