from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from autoPyTorch.pipeline.components.preprocessing.scaling.base_scaler import BaseScaler


class StandardScaler(BaseScaler):
    """
    Standardise numerical columns/features by removing mean and scaling to unit/variance
    """
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 with_mean: bool = True,
                 with_std: bool = True):
        super().__init__(random_state)
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:
        self.preprocessor = SklearnStandardScaler(with_mean=self.with_mean, with_std=self.with_std, copy=False)
        self.column_transformer = make_column_transformer((self.preprocessor, X['numerical_columns']),
                                                          remainder='passthrough')
        self.column_transformer.fit(X['train'])  # TODO read data from local file.
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        with_mean = CategoricalHyperparameter("with_mean", [True, False], default_value=True)
        with_std = CategoricalHyperparameter("with_std", [True, False], default_value=True)
        cs.add_hyperparameters([with_mean, with_std])
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'StandardScaler',
            'name': 'Standard Scaler',
        }
