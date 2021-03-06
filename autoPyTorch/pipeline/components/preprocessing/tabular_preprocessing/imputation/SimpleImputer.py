from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from sklearn.impute import SimpleImputer as SklearnSimpleImputer

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.base_imputer import BaseImputer


class SimpleImputer(BaseImputer):
    """
    Impute missing values for categorical columns with '!missing!'
    """

    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 numerical_strategy: str = 'mean',
                 categorical_strategy: str = 'most_frequent'):
        super().__init__()
        self.random_state = random_state
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImputer:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """
        self.check_requirements(X, y)
        if len(X['categorical_columns']) != 0:
            if self.categorical_strategy == 'constant_!missing!':
                self.preprocessor['categorical'] = SklearnSimpleImputer(strategy='constant',
                                                                        fill_value='!missing!',
                                                                        copy=False)
            else:
                self.preprocessor['categorical'] = SklearnSimpleImputer(strategy=self.categorical_strategy,
                                                                        copy=False)
        if len(X['numerical_columns']) != 0:
            if self.numerical_strategy == 'constant_zero':
                self.preprocessor['numerical'] = SklearnSimpleImputer(strategy='constant',
                                                                      fill_value=0,
                                                                      copy=False)
            else:
                self.preprocessor['numerical'] = SklearnSimpleImputer(strategy=self.numerical_strategy, copy=False)

        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        assert dataset_properties is not None, "To create hyperparameter search space" \
                                               ", dataset_properties should not be None"
        if len(dataset_properties['numerical_columns']):
            numerical_strategy = CategoricalHyperparameter("numerical_strategy",
                                                           ["mean", "median", "most_frequent", "constant_zero"],
                                                           default_value="mean")
            cs.add_hyperparameter(numerical_strategy)

        if len(dataset_properties['categorical_columns']):
            categorical_strategy = CategoricalHyperparameter("categorical_strategy",
                                                             ["most_frequent", "constant_!missing!"],
                                                             default_value="most_frequent")
            cs.add_hyperparameter(categorical_strategy)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'SimpleImputer',
            'name': 'Simple Imputer',
            'handles_sparse': True
        }
