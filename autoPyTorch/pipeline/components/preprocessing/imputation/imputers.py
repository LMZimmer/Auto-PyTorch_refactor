import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sklearn.impute import SimpleImputer
import numpy as np
from typing import Optional

from autoPyTorch.pipeline.components.preprocessing.imputation import BaseImputer


class CategoricalImputer(BaseImputer):
    '''
    Impute missing values for categorical columns with {FILL_VALUE}
    '''
    FILL_VALUE = 2

    def __init__(self, random_state: int = None):
        self.preprocessor = SimpleImputer(strategy='constant', fill_value=CategoricalImputer.FILL_VALUE, copy=False)
        self.random_state = random_state


class NumericalImputer(BaseImputer):
    '''
    Impute missing values for numerical columns using one of {'mean', 'median', 'most_frequent', 'constant_zero'} strategy.
    '''
    def __init__(self, random_state: int = None, strategy: str = 'mean'):
        self.random_state = random_state
        if strategy == 'constant_zero':
            self.preprocessor = SimpleImputer(strategy='constant', fill_value=0, copy=False)
        else:
            self.preprocessor = SimpleImputer(strategy=strategy, copy=False)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[dict] = None) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        strategy = CSH.CategoricalHyperparameter("strategy", ["mean", "median", "most_frequent", "constant_zero"], default_value="mean")
        cs.add_hyperparameter(strategy)
        return cs
