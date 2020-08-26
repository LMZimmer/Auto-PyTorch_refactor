import copy
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sklearn.base import clone

from autoPyTorch.pipeline.components.preprocessing.scaling.base_scaler_choice import ScalerChoice
from autoPyTorch.pipeline.components.preprocessing.scaling.MinMaxScaler import MinMaxScaler
from autoPyTorch.pipeline.components.preprocessing.scaling.NoneScaler import NoneScaler
from autoPyTorch.pipeline.components.preprocessing.scaling.Normalizer import Normalizer
from autoPyTorch.pipeline.components.preprocessing.scaling.StandardScaler import StandardScaler


class TestNormalizer(unittest.TestCase):

    def test_get_config_space(self):
        config = Normalizer.get_hyperparameter_search_space().sample_configuration()
        estimator = Normalizer(**config)
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

    def test_l2_norm(self):
        data = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [4, 5, 6],
                      [11, 12, 13],
                      [17, 18, 19],
                      [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        scaler_component = Normalizer(norm='l2')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if scaler added to X is instance of self
        self.assertEqual(X['scaler'], scaler_component)

        transformed = scaler_component(data[test_indices])

        assert_allclose(transformed, np.array([[0.50257071, 0.57436653, 0.64616234],
                                         [0.54471514, 0.5767572, 0.60879927],
                                         [0.5280169, 0.57601843, 0.62401997]]))

    def test_l1_norm(self):
        data = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [4, 5, 6],
                      [11, 12, 13],
                      [17, 18, 19],
                      [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        scaler_component = Normalizer(norm='l1')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if scaler added to X is instance of self
        self.assertEqual(X['scaler'], scaler_component)
        transformed = scaler_component(data[test_indices])

        assert_allclose(transformed, np.array([[0.29166667, 0.33333333, 0.375],
                                          [0.31481481, 0.33333333, 0.35185185],
                                          [0.30555556, 0.33333333, 0.36111111]]))

    def test_max_norm(self):
        data = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [4, 5, 6],
                      [11, 12, 13],
                      [17, 18, 19],
                      [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        scaler_component = Normalizer(norm='max')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if scaler added to X is instance of self
        self.assertEqual(X['scaler'], scaler_component)
        transformed = scaler_component(data[test_indices])

        assert_allclose(transformed, np.array([[0.77777778, 0.88888889, 1],
                                          [0.89473684, 0.94736842, 1],
                                          [0.84615385, 0.92307692, 1]]))

    def test_l2_norm(self):
        data = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [4, 5, 6],
                      [11, 12, 13],
                      [17, 18, 19],
                      [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        scaler_component = Normalizer(norm='l2')

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if scaler added to X is instance of self
        self.assertEqual(X['scaler'], scaler_component)
        transformed = scaler_component(data[test_indices])

        assert_allclose(transformed, np.array([[0.50257071, 0.57436653, 0.64616234],
                                         [0.54471514, 0.5767572, 0.60879927],
                                         [0.5280169, 0.57601843, 0.62401997]]))


class TestMinMaxScaler(unittest.TestCase):

    def test_minmax_scaler(self):
        data = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [4, 5, 6],
                      [11, 12, 13],
                      [17, 18, 19],
                      [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        scaler_component = MinMaxScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if scaler added to X is instance of self
        self.assertEqual(X['scaler'], scaler_component)

        transformed = scaler_component(data[test_indices])
        assert_allclose(transformed, np.array([[0.46153846, 0.46153846, 0.46153846],
                                          [1.23076923, 1.23076923, 1.23076923],
                                          [0.76923077, 0.76923077, 0.76923077]]))


class TestStandardScaler(unittest.TestCase):

    def test_standard_scaler(self):
        data = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [4, 5, 6],
                      [11, 12, 13],
                      [17, 18, 19],
                      [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        scaler_component = StandardScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if scaler added to X is instance of self
        self.assertEqual(X['scaler'], scaler_component)

        transformed = scaler_component(data[test_indices])

        assert_allclose(transformed, np.array([[0.11995203, 0.11995203, 0.11995203],
                                          [1.91923246, 1.91923246, 1.91923246],
                                          [0.8396642, 0.8396642, 0.8396642]]))


class TestNoneScaler(unittest.TestCase):

    def test_none_scaler(self):
        data = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [4, 5, 6],
                      [11, 12, 13],
                      [17, 18, 19],
                      [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        categorical_columns = list()
        numerical_columns = [0, 1, 2]
        X = {
            'train': data[train_indices],
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
        }
        scaler_component = NoneScaler()

        scaler_component = scaler_component.fit(X)
        X = scaler_component.transform(X)

        # check if scaler added to X is instance of self
        self.assertEqual(X['scaler'], scaler_component)
        transformed = scaler_component(data[test_indices])

        assert_allclose(transformed, transformed)


class TestRescalerChoice(unittest.TestCase):

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        rescaler_choice = ScalerChoice()
        cs = rescaler_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(rescaler_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            rescaler_choice.set_hyperparameters(config)

            self.assertEqual(rescaler_choice.choice.__class__,
                             rescaler_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(rescaler_choice.choice))
                self.assertEqual(value, rescaler_choice.choice.__dict__[key])
