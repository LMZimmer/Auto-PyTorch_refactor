from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import get_preprocess_transforms, preprocess
from autoPyTorch.utils.common import FitRequirement


class EarlyPreprocessing(autoPyTorchSetupComponent):

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super().__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('is_small_preprocess', (bool,), user_defined=True),
            FitRequirement('X_train', (np.ndarray, pd.DataFrame, csr_matrix), user_defined=True),
            FitRequirement('train_indices', (List,), user_defined=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> "EarlyPreprocessing":
        self.check_requirements(X, y)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        transforms = get_preprocess_transforms(X)

        if X['is_small_preprocess']:
            X['X_train'] = preprocess(dataset=X['X_train'], transforms=transforms,
                                      indices=X['train_indices'])
            if 'X_test' in X:
                X['X_test'] = preprocess(dataset=X['X_test'], transforms=transforms)
        else:
            X.update({'preprocess_transforms': transforms})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:
        return ConfigurationSpace()

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'EarlyPreprocessing',
            'name': 'Early Preprocessing Node',
        }

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
