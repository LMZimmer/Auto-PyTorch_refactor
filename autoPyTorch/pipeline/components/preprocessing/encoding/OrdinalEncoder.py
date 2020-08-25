from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder as OE

from autoPyTorch.pipeline.components.preprocessing.encoding import BaseEncoder


class OrdinalEncoder(BaseEncoder):
    """
    Encode categorical features as a one-hot numerical array
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.random_state = random_state
        self.preprocessor: Optional[BaseEstimator] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEncoder:
        self.preprocessor = OE(categories='auto')
        self.preprocessor.fit(X['train'])  # TODO read data from local file.
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'OrdinalEncoder',
            'name': 'Ordinal Encoder',
        }
