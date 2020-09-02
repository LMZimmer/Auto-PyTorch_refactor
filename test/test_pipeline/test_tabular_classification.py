import unittest
import unittest.mock

import numpy as np

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class PipelineTest(unittest.TestCase):

    def test_pipeline_fit(self):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""
        number_features = np.random.randint(low=5, high=15, size=20)
        number_classes = np.random.randint(low=2, high=8, size=20)
        number_datapoints = np.random.randint(low=4, high=10, size=20)
        for num_features, num_classes, num_datapoints in zip(number_features, number_classes, number_datapoints):
            train_data = np.random.random((num_datapoints, num_features))
            dataset_properties = {'numerical_columns': list(range(num_features)), 'categorical_columns': []}
            pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
            cs = pipeline.get_hyperparameter_search_space()
            config = cs.sample_configuration()
            pipeline.set_hyperparameters(config)
            print(config)
            pipeline.fit(
                {'num_features': num_features,
                 'num_classes': num_classes,
                 'numerical_columns': list(range(num_features)),
                 'categorical_columns': [],
                 'train': train_data}
            )

    def test_default_configuration(self):
        """Makes sure that when no config is set, we can trust the
        default configuration from the space"""
        num_features = 4
        num_classes = 2
        num_datapoints = 4
        train_data = np.random.random((num_datapoints, num_features))
        dataset_properties = {'numerical_columns': list(range(num_features)), 'categorical_columns': []}
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)

        pipeline.fit(
            {'num_features': num_features,
             'num_classes': num_classes,
             'train': train_data,
             'numerical_columns': list(range(num_features)),
             'categorical_columns': []}
        )

    def test_network_optimizer_lr_handshake(self):
        """Fitting a network should put the network in the X"""

        # Create the pipeline to check. A random config should be sufficient
        dataset_properties = {'numerical_columns': [], 'categorical_columns': []}
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        # Make sure that fitting a network adds a "network" to X
        self.assertIn('network', pipeline.named_steps.keys())
        X = pipeline.named_steps['network'].fit(
            {'num_features': 10, 'num_classes': 2},
            None
        ).transform({})
        self.assertIn('network', X)

        # Then fitting a optimizer should fail if no network:
        self.assertIn('optimizer', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError, 'Could not parse the network'):
            pipeline.named_steps['optimizer'].fit({}, None)

        # No error when network is passed
        X = pipeline.named_steps['optimizer'].fit(X, None).transform(X)
        self.assertIn('optimizer', X)

        # Then fitting a optimizer should fail if no network:
        self.assertIn('lr_scheduler', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError,
                                    'the fit dictionary Must contain a valid optimizer'):
            pipeline.named_steps['lr_scheduler'].fit({}, None)

        # No error when network is passed
        X = pipeline.named_steps['lr_scheduler'].fit(X, None).transform(X)
        self.assertIn('optimizer', X)


if __name__ == '__main__':
    unittest.main()
