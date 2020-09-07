from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from autoPyTorch.pipeline.components.preprocessing.scaling.base_scaler import BaseScaler


class MinMaxScaler(BaseScaler):
    """
    Scale numerical columns/features into feature_range
    """
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 feature_range: Tuple[Union[int, float], Union[int, float]] = (0, 1)):
        self.random_state = random_state
        self.feature_range = feature_range

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)

        self.preprocessor = SklearnMinMaxScaler(feature_range=self.feature_range, copy=False)
        self.column_transformer = make_column_transformer((self.preprocessor, X['numerical_columns']),
                                                          remainder='passthrough')
        self.column_transformer.fit(X['train'])  # TODO read data from local file.
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'MinMaxScaler',
            'name': 'MinMaxScaler',
        }
