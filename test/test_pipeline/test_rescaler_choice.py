import unittest
import numpy as np
from numpy.testing import assert_allclose
from ConfigSpace.configuration_space import ConfigurationSpace
from autoPyTorch.pipeline.components.preprocessing.rescaling import RescalerChoice


class TestRescalerChoice(unittest.TestCase):
    
    def test_choice(self):
        choice = RescalerChoice()
        try:
            configuration = choice.get_hyperparameter_search_space().sample_configuration()
            self.assertIsInstance(configuration, ConfigurationSpace)
        except:
            self.assertRaises(ValueError)
       
