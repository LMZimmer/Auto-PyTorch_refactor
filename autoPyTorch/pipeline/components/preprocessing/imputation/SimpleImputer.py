from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer as SklearnSimpleImputer

import torch

from autoPyTorch.pipeline.components.preprocessing.imputation.base_imputer import BaseImputer


class SimpleImputer(BaseImputer):
    """
    Impute missing values for categorical columns with '!missing!'
    """

    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 numerical_strategy: str = 'mean',
                 categorical_strategy: str = 'constant_!missing!'):
        self.random_state = random_state
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.preprocessor: Dict[str, BaseEstimator] = dict()

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
        if self.categorical_strategy == 'constant_!missing!':
            self.preprocessor['categorical'] = SklearnSimpleImputer(strategy='constant',
                                                                    fill_value='!missing!',
                                                                    copy=False)
        else:
            self.preprocessor['categorical'] = SklearnSimpleImputer(strategy=self.categorical_strategy,
                                                                    copy=False)

        if self.numerical_strategy == 'constant_zero':
            self.preprocessor['numerical'] = SklearnSimpleImputer(strategy='constant',
                                                                  fill_value=0,
                                                                  copy=False)
        else:
            self.preprocessor['numerical'] = SklearnSimpleImputer(strategy=self.numerical_strategy, copy=False)

        if len(X['categorical_columns']) == 0:
            self.column_transformer = make_column_transformer(
                (self.preprocessor['numerical'], X['numerical_columns']),
                remainder='passthrough')
        elif len(X['numerical_columns']) == 0:
            self.column_transformer = make_column_transformer(
                (self.preprocessor['categorical'], X['categorical_columns']),
                remainder='passthrough')
        else:
            self.column_transformer = make_column_transformer(
                (self.preprocessor['categorical'], X['categorical_columns']),
                (self.preprocessor['numerical'], X['numerical_columns']),
                remainder='passthrough')

        self.column_transformer.fit(X['train'].astype(object))  # TODO read data from local file.
        return self

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        """
        Makes the autoPyTorchPreprocessingComponent Callable. Calling the component
        calls the transform function of the underlying preprocessor and
        returns the transformed array.
        Args:
            X (Union[np.ndarray, torch.tensor]): input data tensor

        Returns:
            Union[np.ndarray, torch.tensor]: Transformed data tensor
        """
        if self.column_transformer is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))
        X = self.column_transformer.transform(X.astype(object))
        return X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        numerical_strategy = CategoricalHyperparameter("numerical_strategy",
                                                       ["mean", "median", "most_frequent", "constant_zero"],
                                                       default_value="mean")
        categorical_strategy = CategoricalHyperparameter("categorical_strategy",
                                                         ["most_frequent", "constant_!missing!"],
                                                         default_value="constant_!missing!")
        cs.add_hyperparameter(numerical_strategy)
        cs.add_hyperparameter(categorical_strategy)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'SimpleImputer',
            'name': 'Simple Imputer',
        }
