from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder as OHE

from autoPyTorch.pipeline.components.preprocessing.encoding.base_encoder import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """
    Encode categorical features as a one-hot numerical array
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEncoder:

        self.check_requirements(X, y)

        self.preprocessor = OHE(categories='auto', sparse=False, handle_unknown='error')
        self.column_transformer = make_column_transformer((self.preprocessor, X['categorical_columns']),
                                                          remainder='passthrough')
        self.column_transformer.fit(X['train'])  # TODO read data from local file.
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'OneHotEncoder',
            'name': 'One Hot Encoder',
        }
