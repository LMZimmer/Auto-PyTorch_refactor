from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from autoPyTorch.pipeline.components.preprocessing.scaling import BaseScaler


class StandardScaler(BaseScaler):
    """
    Standardise numerical columns/features by removing mean and scaling to unit/variance
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None,
                 with_mean: bool = True,
                 with_std: bool = True):
        self.random_state = random_state
        self.with_mean = with_mean
        self.with_std = with_std
        self.preprocessor: Optional[BaseEstimator] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> BaseScaler:
        self.preprocessor = SklearnStandardScaler(with_mean=self.with_mean, with_std=self.with_std, copy=False)
        self.preprocessor.fit(X, y)
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
