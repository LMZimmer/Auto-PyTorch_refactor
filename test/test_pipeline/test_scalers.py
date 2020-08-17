import copy
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sklearn.base import clone

from autoPyTorch.pipeline.components.preprocessing.scaling import ScalerChoice
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
        X = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6],
                     [11, 12, 13], [17, 18, 19], [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])

        preprocessor = Normalizer(norm='l2')

        preprocessor = preprocessor.fit(X[train_indices])
        X_train = preprocessor.transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])

        assert_allclose(X_train, np.array([[0.26726124, 0.53452248, 0.80178373],
                                          [0.45584231, 0.56980288, 0.68376346],
                                          [0.53806371, 0.57649683, 0.61492995]]))
        assert_allclose(X_test, np.array([[0.50257071, 0.57436653, 0.64616234],
                                         [0.54471514, 0.5767572, 0.60879927],
                                         [0.5280169, 0.57601843, 0.62401997]]))

    def test_l1_norm(self):
        X = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6],
                     [11, 12, 13], [17, 18, 19], [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])

        preprocessor = Normalizer(norm='l1')

        preprocessor = preprocessor.fit(X[train_indices])
        X_train = preprocessor.transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])

        assert_allclose(X_train, np.array([[0.16666667, 0.33333333, 0.5],
                                          [0.26666667, 0.33333333, 0.4],
                                          [0.31111111, 0.33333333, 0.35555556]]))
        assert_allclose(X_test, np.array([[0.29166667, 0.33333333, 0.375],
                                         [0.31481481, 0.33333333, 0.35185185],
                                         [0.30555556, 0.33333333, 0.36111111]]))

    def test_max_norm(self):
        X = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6],
                     [11, 12, 13], [17, 18, 19], [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])

        preprocessor = Normalizer(norm='max')

        preprocessor = preprocessor.fit(X[train_indices])
        X_train = preprocessor.transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])

        assert_allclose(X_train, np.array([[0.33333333, 0.66666667, 1],
                                          [0.66666667, 0.83333333, 1],
                                          [0.875, 0.9375, 1]]))
        assert_allclose(X_test, np.array([[0.77777778, 0.88888889, 1],
                                          [0.89473684, 0.94736842, 1],
                                          [0.84615385, 0.92307692, 1]]))


class TestMinMaxScaler(unittest.TestCase):

    def test_minmax_scaler(self):
        X = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6],
                     [11, 12, 13], [17, 18, 19], [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])

        preprocessor = MinMaxScaler()

        preprocessor = preprocessor.fit(X[train_indices])
        X_train = preprocessor.transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])

        assert_allclose(X_train, np.array([[0, 0, 0],
                                          [0.23076923, 0.23076923, 0.23076923],
                                          [1, 1, 1]]))

        assert_allclose(X_test, np.array([[0.46153846, 0.46153846, 0.46153846],
                                         [1.23076923, 1.23076923, 1.23076923],
                                         [0.76923077, 0.76923077, 0.76923077]]))


class TestStandardScaler(unittest.TestCase):

    def test_standard_scaler(self):
        X = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6],
                     [11, 12, 13], [17, 18, 19],
                     [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])

        preprocessor = StandardScaler()

        preprocessor = preprocessor.fit(X[train_indices])
        X_train = preprocessor.transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])

        assert_allclose(X_train, np.array([[-0.95961623, -0.95961623, -0.95961623],
                                          [-0.4198321, -0.4198321, -0.4198321],
                                          [1.37944833, 1.37944833, 1.37944833]]))

        assert_allclose(X_test, np.array([[0.11995203, 0.11995203, 0.11995203],
                                         [1.91923246, 1.91923246, 1.91923246],
                                         [0.8396642, 0.8396642, 0.8396642]]))

class TestNoneScaler(unittest.TestCase):

    def test_none_scaler(self):
        X = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6],
                     [11, 12, 13], [17, 18, 19],
                     [14, 15, 16]])
        train_indices = np.array([0, 2, 5])
        test_indices = np.array([1, 4, 3])
        X_train = X[train_indices]
        X_test = X[test_indices]

        preprocessor = NoneScaler()

        preprocessor = preprocessor.fit(X[train_indices])
        X_train_trans = preprocessor.transform(X[train_indices])
        X_test_trans = preprocessor.transform(X[test_indices])

        assert_allclose(X_train, X_train_trans)

        assert_allclose(X_test, X_test_trans)


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
