import unittest
import unittest.mock

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import (
    FeatureDataLoader
)


class TestFeatureDataLoader(unittest.TestCase):
    def test_build_transform_small_preprocess_true(self):
        """
        Makes sure a proper composition is created
        """
        loader = FeatureDataLoader()

        fit_dictionary = {'is_small_preprocess': True}
        for thing in ['imputer', 'scaler', 'encoder']:
            fit_dictionary[thing] = unittest.mock.Mock()

        compose = loader.build_transform(fit_dictionary)

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        # No preprocessing needed here as it was done before
        self.assertEqual(len(compose.transforms), 1)

    def test_build_transform_small_preprocess_false(self):
        """
        Makes sure a proper composition is created
        """
        loader = FeatureDataLoader()

        fit_dictionary = {'is_small_preprocess': False}
        fit_dictionary['preprocess_transforms'] = unittest.mock.Mock()

        compose = loader.build_transform(fit_dictionary)

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        # We expect the to tensor and the preproces transforms
        self.assertEqual(len(compose.transforms), 2)
