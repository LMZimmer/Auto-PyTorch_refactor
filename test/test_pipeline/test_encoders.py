import unittest

import numpy as np
from numpy.testing import assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.encoding.NoneEncoder import NoneEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OneHotEncoder import OneHotEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OrdinalEncoder import OrdinalEncoder


class TestEncoders(unittest.TestCase):

    def test_one_hot_encoder_no_unknown(self):
        X = np.array([[1, 'male'],
                     [1, 'female'],
                     [3, 'female'],
                     [2, 'male'],
                     [2, 'female']])

        categorical_features = [False, True]
        X = X[:, categorical_features]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4])
        preprocessor = OneHotEncoder()
        preprocessor.fit(X[train_indices])
        X_transformed = preprocessor.transform(X[test_indices])
        preprocessor = preprocessor.get_preprocessor()
        categories = preprocessor.categories_[0].tolist()
        expected_categories = ['male', 'female']

        self.assertCountEqual(categories, expected_categories)
        assert_array_equal(X_transformed, [[1, 0], [1, 0]])

        def test_one_hot_encoder_with_unknown(self):
            X = np.array([[1, 'male'],
                         [1, 'female'],
                         [3, 'female'],
                         [2, 'male'],
                         [2, 'unknown']])

            categorical_features = [False, True]
            X = X[:, categorical_features].astype(object)
            train_indices = np.array([0, 2, 3])
            test_indices = np.array([1, 4])
            preprocessor = OneHotEncoder()
            preprocessor.fit(X[train_indices])
            try:
                preprocessor.transform(X[test_indices])
            except ValueError as msg:
                self.assertRegex(msg, r'^Found unknown categories \[.+\] in column [0-9]+ during \
                                        transform in autoPyTorch\.pipeline\.components.+$')

    def test_ordinal_encoder(self):
        X = np.array([[1, 'male'],
                     [1, 'female'],
                     [3, 'unknown'],
                     [2, 'female'],
                     [2, 'female']])

        categorical_features = [False, True]
        X = X[:, categorical_features]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4])
        preprocessor = OrdinalEncoder()
        preprocessor.fit(X[train_indices])
        X_transformed = preprocessor.transform(X[test_indices])
        preprocessor = preprocessor.get_preprocessor()
        categories = preprocessor.categories_[0].tolist()
        expected_categories = ['male', 'female', 'unknown']

        self.assertCountEqual(categories, expected_categories)

        assert_array_equal(X_transformed, [[0], [0]])

    def test_none_encoder(self):
        X = np.array([[1, 'male'],
                     [1, 'female'],
                     [3, 'unknown'],
                     [2, 'female'],
                     [2, 'female']])

        categorical_features = [False, True]
        X = X[:, categorical_features]
        train_indices = np.array([0, 2, 3])
        preprocessor = NoneEncoder()
        preprocessor.fit(X[train_indices])
        X_transformed = preprocessor.transform(X[train_indices])

        assert_array_equal(X_transformed, X[train_indices])
