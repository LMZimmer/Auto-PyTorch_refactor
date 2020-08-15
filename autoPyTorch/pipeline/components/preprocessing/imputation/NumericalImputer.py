from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from sklearn.impute import SimpleImputer

from autoPyTorch.pipeline.components.preprocessing.imputation.base_imputer import BaseImputer


class NumericalImputer(BaseImputer):
    '''
    Impute missing values for numerical columns using
    one of {'mean', 'median', 'most_frequent', 'constant_zero'} strategy.
    '''
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None, strategy: str = 'mean'):
        self.random_state = random_state
        self.strategy = strategy
        if strategy == 'constant_zero':
            self.preprocessor = SimpleImputer(strategy='constant', fill_value=0, copy=False)
        else:
            self.preprocessor = SimpleImputer(strategy=self.strategy, copy=False)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        strategy = CategoricalHyperparameter("strategy",
                                             ["mean", "median", "most_frequent", "constant_zero"],
                                             default_value="mean")
        cs.add_hyperparameter(strategy)
        return cs
