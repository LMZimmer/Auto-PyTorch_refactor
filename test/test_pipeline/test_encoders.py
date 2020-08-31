import unittest

import numpy as np
from numpy.testing import assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.encoding.NoEncoder import NoEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OneHotEncoder import OneHotEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OrdinalEncoder import OrdinalEncoder


class TestEncoders(unittest.TestCase):

    def test_one_hot_encoder_no_unknown(self):
        data = np.array([[1, 'male'],
                         [1, 'female'],
                         [3, 'female'],
                         [2, 'male'],
                         [2, 'female']])

        categorical_columns = [1]
        numerical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        encoder_component = OneHotEncoder()
        encoder_component.fit(X)
        X = encoder_component.transform(X)
        # check if encoder added to X is instance of self
        self.assertEqual(X['encoder'], encoder_component)

        transformed = encoder_component(data[test_indices])
        # check if the transform is correct
        assert_array_equal(transformed, [['1.0', '0.0', 1], ['1.0', '0.0', 2]])

    def test_one_hot_encoder_with_unknown(self):
        data = np.array([[1, 'male'],
                         [1, 'female'],
                         [3, 'female'],
                         [2, 'male'],
                         [2, 'female']])

        categorical_columns = [1]
        numerical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        encoder_component = OneHotEncoder()
        encoder_component.fit(X)
        X = encoder_component.transform(X)

        # check if encoder added to X is instance of self
        self.assertEqual(X['encoder'], encoder_component)
        try:
            encoder_component(data[test_indices])
        except ValueError as msg:
            self.assertRegex(str(msg), r'Found unknown categories .+?in column [0-9]+ during transform in <class '
                                       r'\'autoPyTorch\.pipeline\.components.+')

    def test_ordinal_encoder(self):

        data = np.array([[1, 'male'],
                         [1, 'female'],
                         [3, 'male'],
                         [2, 'female'],
                         [2, 'male']])

        categorical_columns = [1]
        numerical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        encoder_component = OrdinalEncoder()
        encoder_component.fit(X)
        X = encoder_component.transform(X)

        # check if encoder added to X is instance of self
        self.assertEqual(X['encoder'], encoder_component)

        transformed = encoder_component(data[test_indices])

        # check if we got the expected transformed array
        assert_array_equal(transformed, [['0.0', 1], ['1.0', 2]])

    def test_none_encoder(self):

        data = np.array([[1, 'male'],
                         [1, 'female'],
                         [3, 'unknown'],
                         [2, 'female'],
                         [2, 'male']])

        categorical_columns = [1]
        numerical_columns = [0]
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1, 4])
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        encoder_component = NoEncoder()
        encoder_component.fit(X)
        _ = encoder_component.transform(X)

        # check if encoder added to X is instance of self
        self.assertEqual(X['encoder'], encoder_component)

        transformed = encoder_component(data[test_indices])

        # check if we got the expected transformed array
        assert_array_equal(transformed, data[test_indices])
