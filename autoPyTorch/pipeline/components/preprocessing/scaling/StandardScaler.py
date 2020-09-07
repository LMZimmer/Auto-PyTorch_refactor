from typing import Any, Dict, Optional, Union

import numpy as np

from scipy.sparse import issparse

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from autoPyTorch.pipeline.components.preprocessing.scaling.base_scaler import BaseScaler


class StandardScaler(BaseScaler):
    """
    Standardise numerical columns/features by removing mean and scaling to unit/variance
    """
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)

        with_mean, with_std = (False, False) if issparse(X['train']) else (True, True)
        self.preprocessor = SklearnStandardScaler(with_mean=with_mean, with_std=with_std, copy=False)
        self.column_transformer = make_column_transformer((self.preprocessor, X['numerical_columns']),
                                                          remainder='passthrough')
        self.column_transformer.fit(X['train'])  # TODO read data from local file.
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'StandardScaler',
            'name': 'Standard Scaler',
        }
