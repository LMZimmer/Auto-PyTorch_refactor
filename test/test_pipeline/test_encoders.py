import copy
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.encoding.NoneEncoder import NoneEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OneHotEncoder import OneHotEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OrdinalEncoder import OrdinalEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.base_encoder_choice import EncoderChoice


class TestEncoders(unittest.TestCase):

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        encoder_choice = EncoderChoice()
        cs = encoder_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(encoder_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            encoder_choice.set_hyperparameters(config)

            self.assertEqual(encoder_choice.choice.__class__,
                             encoder_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(encoder_choice.choice))
                self.assertEqual(value, encoder_choice.choice.__dict__[key])

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
        encoder_component = NoneEncoder()
        encoder_component.fit(X)
        _ = encoder_component.transform(X)

        # check if encoder added to X is instance of self
        self.assertEqual(X['encoder'], encoder_component)

        transformed = encoder_component(data[test_indices])

        # check if we got the expected transformed array
        assert_array_equal(transformed, data[test_indices])
