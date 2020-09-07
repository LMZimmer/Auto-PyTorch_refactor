from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import Normalizer as SklearnNormalizer

from autoPyTorch.pipeline.components.preprocessing.scaling.base_scaler import BaseScaler


class Normalizer(BaseScaler):
    """
    Normalises samples individually according to norm {mean_abs, mean_squared, max}
    """

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None, norm: str = 'mean_squared'):
        """
        Args:
            random_state (Optional[Union[np.random.RandomState, int]]): Determines random number generation for
            subsampling and smoothing noise.
            norm (str): {mean_abs, mean_squared, max} default: mean_squared
        """
        self.random_state = random_state
        self.norm = norm

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)

        map_norm = dict({"mean_abs": "l1", "mean_squared": "l2", "max": "max"})
        self.preprocessor = SklearnNormalizer(norm=map_norm[self.norm], copy=False)
        self.column_transformer = make_column_transformer((self.preprocessor, X["numerical_columns"]),
                                                          remainder='passthrough')
        self.column_transformer.fit(X['train'])  # TODO read data from local file.
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        norm = CategoricalHyperparameter("norm", ["mean_abs", "mean_squared", "max"], default_value="mean_squared")
        cs.add_hyperparameter(norm)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'Normalizer',
            'name': 'Normalizer',
        }
