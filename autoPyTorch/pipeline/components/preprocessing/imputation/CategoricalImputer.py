from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer

import torch

from autoPyTorch.pipeline.components.preprocessing.imputation.base_imputer import BaseImputer


class CategoricalImputer(BaseImputer):
    '''
    Impute missing values for categorical columns with '!missing!'
    '''
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.random_state = random_state
        self.preprocessor: Optional[BaseEstimator] = None

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
        self.preprocessor = SimpleImputer(strategy='constant', fill_value='!missing!', missing_values='nan', copy=False)
        self.preprocessor.fit(X['train'].astype(object))  # TODO read data from local file.
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
        if self.preprocessor is None:
            raise ValueError("cant call {} without fitting the preprocessor first.".format(self.__class__.__name__))
        X = self.preprocessor.transform(X.astype(object))
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'CategoricalImputer',
            'name': 'Categorical Imputer',
        }
