from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from sklearn.preprocessing import Normalizer as SklearnNormalizer

from autoPyTorch.pipeline.components.preprocessing.scaling import BaseScaler


class Normalizer(BaseScaler):
    """
    Normalises the columns/features using according to norm {l1, l2, max}
    """

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None, norm: str = 'l2'):
        self.random_state = random_state
        self.norm = norm
        self.preprocessor = SklearnNormalizer(norm=self.norm, copy=False)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        norm = CategoricalHyperparameter("norm", ["l1", "l2", "max"], default_value="l2")
        cs.add_hyperparameter(norm)
        return cs
