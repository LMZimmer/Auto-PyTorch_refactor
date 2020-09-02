import unittest
import unittest.mock

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class DummyComponent(autoPyTorchComponent):
    def __init__(self, a=0, b='orange', random_state=None):
        self.a = a
        self.b = b
        self.fitted = False

    def get_hyperparameter_search_space(self, dataset_properties=None):
        cs = CS.ConfigurationSpace()
        a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False)
        b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue'])
        cs.add_hyperparameters([a, b])
        return cs

    def fit(self, X, y):
        self.fitted = True
        return self


class DummyChoice(autoPyTorchChoice):
    def get_components(self):
        return {
            'DummyComponent2': DummyComponent,
            'DummyComponent3': DummyComponent,
        }

    def get_hyperparameter_search_space(self, dataset_properties=None, default=None,
                                        include=None, exclude=None):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CSH.CategoricalHyperparameter(
                '__choice__',
                list(self.get_components().keys()),
            )
        )
        return cs


class BasePipelineMock(BasePipeline):
    def __init__(self):
        pass

    def _get_pipeline_steps(self, dataset_properties):
        return [
            ('DummyComponent1', DummyComponent(a=10, b='red')),
            ('DummyChoice', DummyChoice(self.dataset_properties))
        ]


class PipelineTest(unittest.TestCase):
    def setUp(self):
        """Create a pipeline and test the different properties of it"""
        self.pipeline = BasePipelineMock()
        self.pipeline.dataset_properties = {}
        self.pipeline.steps = [
            ('DummyComponent1', DummyComponent(a=10, b='red')),
            ('DummyChoice', DummyChoice(self.pipeline.dataset_properties))
        ]

    def test_pipeline_base_config_space(self):
        """Makes sure that the pipeline can build a proper
        configuration space via its base config methods"""
        cs = self.pipeline._get_base_search_space(
            cs=CS.ConfigurationSpace(),
            include={}, exclude={}, dataset_properties={},
            pipeline=self.pipeline.steps
        )

        # The hyperparameters a and b of the dummy component
        # must be in the hyperparameter search space
        # If parsing the configuration correctly, hyper param a
        # lower bound should be properly defined
        self.assertIn('DummyComponent1:a', cs)
        self.assertEqual(10,
                         cs.get_hyperparameter('DummyComponent1:a').lower)
        self.assertIn('DummyComponent1:b', cs)

        # For the choice, we make sure the choice
        # is among components 2 and 4
        self.assertIn('DummyChoice:__choice__', cs)
        self.assertEqual(('DummyComponent2', 'DummyComponent3'),
                         cs.get_hyperparameter('DummyChoice:__choice__').choices)

    def test_pipeline_set_config(self):
        config = self.pipeline._get_base_search_space(
            cs=CS.ConfigurationSpace(),
            include={}, exclude={}, dataset_properties={},
            pipeline=self.pipeline.steps
        ).sample_configuration()

        self.pipeline.set_hyperparameters(config)

        # Check that the proper hyperparameters where set
        config_dict = config.get_dictionary()
        self.assertEqual(config_dict['DummyComponent1:a'],
                         self.pipeline.named_steps['DummyComponent1'].a)
        self.assertEqual(config_dict['DummyComponent1:b'],
                         self.pipeline.named_steps['DummyComponent1'].b)

        # Make sure that the proper component choice was made
        # according to the config
        # The orange check makes sure that the pipeline is setting the
        # hyperparameters individually, as orange should only happen on the
        # choice, as it is not a hyperparameter from the cs
        self.assertIsInstance(self.pipeline.named_steps['DummyChoice'].choice, DummyComponent)
        self.assertEqual('orange',
                         self.pipeline.named_steps['DummyChoice'].choice.b)

    @unittest.skip("Fit has not been properly coded")
    def test_pipeline_fit(self):
        """Make sure that the pipeline is able to fit every step properly"""
        pass

    @unittest.skip("Predict has not been properly coded")
    def test_pipeline_predict(self):
        pass


if __name__ == '__main__':
    unittest.main()
