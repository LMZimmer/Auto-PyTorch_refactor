import unittest

import numpy as np
from numpy.testing import assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.imputation.CategoricalImputer import CategoricalImputer
from autoPyTorch.pipeline.components.preprocessing.imputation.NumericalImputer import NumericalImputer


class TestNumericalImputer(unittest.TestCase):

    def test_mean_imputation(self):
        X = np.array([[1, np.nan, 3], [np.nan, 8, 9], [4, 5, np.nan],
                     [np.nan, 2, 3], [7, np.nan, 9], [4, np.nan, np.nan]])
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])

        preprocessor = NumericalImputer(strategy='mean')

        preprocessor = preprocessor.fit(X[train_indices])
        X_train = preprocessor.transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])

        assert_array_equal(X_train, np.array([[1, 3.5, 3], [4, 5, 3], [2.5, 2, 3]]))
        assert_array_equal(X_test, np.array([[2.5, 8, 9], [7, 3.5, 9], [4, 3.5, 3]]))

    # def test_median_imputation(self):
    #     X = np.array([[1, np.nan, 3], [np.nan, 8, 9], [4, 5, np.nan],
    #                  [np.nan, 2, 3], [7, np.nan, 9], [4, np.nan, np.nan]])
    #     train_indices = np.array([0, 2, 3])
    #     test_indices = np.array([1, 4, 5])

    #     preprocessor = NumericalImputer(strategy='median')

    #     preprocessor = preprocessor.fit(X[train_indices])
    #     X_train = preprocessor.transform(X[train_indices])
    #     X_test = preprocessor.transform(X[test_indices])

    #     assert_array_equal(X_train, np.array([[1, 3.5, 3], [4, 5, 3], [2.5, 2, 3]]))
    #     assert_array_equal(X_test, np.array([[2.5, 8, 9], [7, 3.5, 9], [4, 3.5, 3]]))

    # def test_frequent_imputation(self):
    #     X = np.array([[1, np.nan, 3], [np.nan, 8, 9], [4, 5, np.nan],
    #                  [np.nan, 2, 3], [7, np.nan, 9], [4, np.nan, np.nan]])

    #     train_indices = np.array([0, 2, 3, 1])
    #     test_indices = np.array([4, 5])
    #     preprocessor = NumericalImputer(strategy='most_frequent')

    #     preprocessor = preprocessor.fit(X[train_indices])
    #     X_train = preprocessor.transform(X[train_indices])
    #     X_test = preprocessor.transform(X[test_indices])

    #     assert_array_equal(X_train, np.array([[1, 2, 3], [4, 5, 3], [1, 2, 3], [1, 8, 9]]))
    #     assert_array_equal(X_test, np.array([[7, 2, 9], [4, 2, 3]]))

    # def test_zero_imputation(self):
    #     X = np.array([[1, np.nan, 3], [np.nan, 8, 9], [4, 5, np.nan],
    #                  [np.nan, 2, 3], [7, np.nan, 9], [4, np.nan, np.nan]])
    #     train_indices = np.array([0, 2, 3])
    #     test_indices = np.array([1, 4, 5])

    #     preprocessor = NumericalImputer(strategy='constant_zero')

    #     preprocessor = preprocessor.fit(X[train_indices])
    #     X_train = preprocessor.transform(X[train_indices])
    #     X_test = preprocessor.transform(X[test_indices])

    #     assert_array_equal(X_train, np.array([[1, 0, 3], [4, 5, 0], [0, 2, 3]]))
    #     assert_array_equal(X_test, np.array([[0, 8, 9], [7, 0, 9], [4, 0, 0]]))


class TestCategoricalImputer(unittest.TestCase):

    def test_imputation(self):
        X = np.array([['1', 'nan', '3'], ['nan', '8', '9'], ['4', '5', 'nan'],
                     ['nan', '2', '3'], ['7', 'nan', '9'], ['4', 'nan', 'nan']])
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4, 5])

        preprocessor = CategoricalImputer()

        preprocessor = preprocessor.fit(X[train_indices])
        X_train = preprocessor.transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])

        assert_array_equal(X_train, np.array([[1, '!missing!', 3],
                                             [4, 5, '!missing!'],
                                             ['!missing!', 2, 3]]).astype(object))
        assert_array_equal(X_test, np.array([['!missing!', 8, 9],
                                            [7, '!missing!', 9],
                                            [4, '!missing!', '!missing!']]).astype(object))


if __name__ == '__main__':
    unittest.main()
