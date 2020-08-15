import copy
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.encoding import EncoderChoice
from autoPyTorch.pipeline.components.preprocessing.encoding.NoneEncoder import NoneEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OneHotEncoder import OneHotEncoder
from autoPyTorch.pipeline.components.preprocessing.encoding.OrdinalEncoder import OrdinalEncoder


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
            self.assertRegex(str(msg), r'Found unknown categories .+?in column [0-9]+ during transform in <class '
                                       r'\'autoPyTorch\.pipeline\.components.+')

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
