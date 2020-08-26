import unittest

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.base import clone

from autoPyTorch.pipeline.components.preprocessing.imputation.SimpleImputer import SimpleImputer


class TestSimpleImputer(unittest.TestCase):

    def test_get_config_space(self):
        config = SimpleImputer.get_hyperparameter_search_space().sample_configuration()
        estimator = SimpleImputer(**config)
        estimator_clone = clone(estimator)
        estimator_clone_params = estimator_clone.get_params()

        # Make sure all keys are copied properly
        for k, v in estimator.get_params().items():
            self.assertIn(k, estimator_clone_params)

        # Make sure the params getter of estimator are honored
        klass = estimator.__class__
        new_object_params = estimator.get_params(deep=False)
        for name, param in new_object_params.items():
            new_object_params[name] = clone(param, safe=False)
        new_object = klass(**new_object_params)
        params_set = new_object.get_params(deep=False)

        for name in new_object_params:
            param1 = new_object_params[name]
            param2 = params_set[name]
            self.assertEqual(param1, param2)

    def test_mean_imputation(self):
        data = np.array([[1, np.nan, 3],
                         [np.nan, 8, 9],
                         [4, 5, np.nan],
                         [np.nan, 2, 3],
                         [7, np.nan, 9],
                         [4, np.nan, np.nan]])
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        imputer_component = SimpleImputer(numerical_strategy='mean')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)

        # check if imputer added to X is instance of self
        self.assertEqual(X['imputer'], imputer_component)

        transformed = imputer_component(data[test_indices])

        assert_array_equal(transformed.astype(str), np.array([['!missing!', 8.0, 9.0],
                                                             [7.0, 3.5, 9.0],
                                                             [4.0, 3.5, 3.0]]))

    def test_median_imputation(self):
        data = np.array([[1, np.nan, 3],
                         [np.nan, 8, 9],
                         [4, 5, np.nan],
                         [np.nan, 2, 3],
                         [7, np.nan, 9],
                         [4, np.nan, np.nan]])
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        imputer_component = SimpleImputer(numerical_strategy='median')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)

        # check if imputer added to X is instance of self
        self.assertEqual(X['imputer'], imputer_component)

        transformed = imputer_component(data[test_indices])

        assert_array_equal(transformed.astype(str), np.array([['!missing!', 8.0, 9.0],
                                                             [7.0, 3.5, 9.0],
                                                             [4.0, 3.5, 3.0]]))

    def test_frequent_imputation(self):
        data = np.array([[1, np.nan, 3],
                         [np.nan, 8, 9],
                         [4, 5, np.nan],
                         [np.nan, 2, 3],
                         [7, np.nan, 9],
                         [4, np.nan, np.nan]])
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        imputer_component = SimpleImputer(numerical_strategy='most_frequent')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)

        # check if imputer added to X is instance of self
        self.assertEqual(X['imputer'], imputer_component)

        transformed = imputer_component(data[test_indices])

        assert_array_equal(transformed.astype(str), np.array([['!missing!', 8.0, 9.0],
                                                             [7.0, 2.0, 9.0],
                                                             [4.0, 2.0, 3.0]]))

    def test_zero_imputation(self):
        data = np.array([[1, np.nan, 3],
                         [np.nan, 8, 9],
                         [4, 5, np.nan],
                         [np.nan, 2, 3],
                         [7, np.nan, 9],
                         [4, np.nan, np.nan]])
        numerical_columns = [1, 2]
        categorical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        imputer_component = SimpleImputer(numerical_strategy='constant_zero')

        imputer_component = imputer_component.fit(X)
        X = imputer_component.transform(X)

        # check if imputer added to X is instance of self
        self.assertEqual(X['imputer'], imputer_component)

        transformed = imputer_component(data[test_indices])

        assert_array_equal(transformed.astype(str), np.array([['!missing!', 8.0, 9.0],
                                                             [7.0, '0', 9.0],
                                                             [4.0, '0', '0']]))


if __name__ == '__main__':
    unittest.main()
