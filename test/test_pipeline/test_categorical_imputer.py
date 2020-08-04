import unittest
import numpy as np
from numpy.testing import assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.imputation.categorical_imputer import CategoricalImputer

class TestCategoricalImputer(unittest.TestCase):
    
    def test_imputation(self):
        X = np.array([[1, np.nan, 3], [np.nan, 8, 9], [4, 5, np.nan],
                [np.nan, 2, 3], [7, np.nan, 9], [4, np.nan, np.nan]])
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])

        imputer = CategoricalImputer()

        imputer = imputer.fit(X[train_indices])
        X_train = imputer.transform(X[train_indices])
        X_test = imputer.transform(X[test_indices])

        assert_array_equal(X_train, np.array([[1, 2, 3], [4, 5, 2], [2, 2, 3]]))
        assert_array_equal(X_test, np.array([[2, 8, 9], [7, 2, 9], [4, 2, 2]]))