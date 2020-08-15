import unittest

import numpy as np
from numpy.testing import assert_allclose

# from autoPyTorch.pipeline.components.preprocessing.rescaling import RescalerChoice
from autoPyTorch.pipeline.components.preprocessing.rescaling.MinMaxScaler import MinMaxScaler
from autoPyTorch.pipeline.components.preprocessing.rescaling.NoneScaler import NoneScaler
from autoPyTorch.pipeline.components.preprocessing.rescaling.Normalizer import Normalizer
from autoPyTorch.pipeline.components.preprocessing.rescaling.StandardScaler import StandardScaler


class TestNormalizer(unittest.TestCase):

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

    def test_minmax_scaler(self):
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


# # class TestRescalerChoice(unittest.TestCase):

# #     def test_choice(self):
# #         choice = RescalerChoice()
# #         try:
# #             configuration = choice.get_hyperparameter_search_space().sample_configuration()
# #             self.assertIsInstance(configuration, ConfigurationSpace)
# #         except:
# #             self.assertRaises(ValueError)
