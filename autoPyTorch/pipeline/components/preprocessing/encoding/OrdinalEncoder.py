from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder as OE

from autoPyTorch.pipeline.components.preprocessing.encoding.base_encoder import BaseEncoder


class OrdinalEncoder(BaseEncoder):
    """
    Encode categorical features as a one-hot numerical array
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(OrdinalEncoder, self).__init__(random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEncoder:

        self.check_requirements(X, y)

        self.preprocessor = OE(categories='auto')
        self.column_transformer = make_column_transformer((self.preprocessor, X['categorical_columns']),
                                                          remainder='passthrough')
        self.column_transformer.fit(X['train'])  # TODO read data from local file.
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'OrdinalEncoder',
            'name': 'Ordinal Encoder',
        }
