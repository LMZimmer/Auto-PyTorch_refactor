import copy
import unittest
import unittest.mock

from ConfigSpace.configuration_space import ConfigurationSpace

from sklearn.base import clone

import torch.nn as nn

import autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice as lr_components
import autoPyTorch.pipeline.components.setup.network.base_network_choice as network_components
import autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice as optimizer_components
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice import (
    BaseLRComponent,
    SchedulerChoice
)
from autoPyTorch.pipeline.components.setup.network.MLPNet import MLPNet
from autoPyTorch.pipeline.components.setup.network.base_network_choice import (
    BaseNetworkComponent,
    NetworkChoice
)
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice import (
    BaseOptimizerComponent,
    OptimizerChoice
)


class DummyLR(BaseLRComponent):
    def __init__(self, random_state=None):
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class DummyOptimizer(BaseOptimizerComponent):
    def __init__(self, random_state=None):
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class DummyNet(BaseNetworkComponent):
    def __init__(self, random_state=None):
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class SchedulerTest(unittest.TestCase):
    def test_every_scheduler_is_valid(self):
        """
        Makes sure that every scheduler is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        scheduler_choice = SchedulerChoice()

        # Make sure all components are returned
        self.assertEqual(len(scheduler_choice.get_components().keys()), 7)

        # For every scheduler in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, scheduler in scheduler_choice.get_components().items():
            config = scheduler.get_hyperparameter_search_space().sample_configuration()
            estimator = scheduler(**config)
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

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the scheduler
        choice"""
        scheduler_choice = SchedulerChoice()
        cs = scheduler_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(scheduler_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            scheduler_choice.set_hyperparameters(config)

            self.assertEqual(scheduler_choice.choice.__class__,
                             scheduler_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(scheduler_choice.choice))
                self.assertEqual(value, scheduler_choice.choice.__dict__[key])

    def test_scheduler_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        self.assertEqual(len(lr_components._addons.components), 0)

        # Then make sure the scheduler can be added and query'ed
        lr_components.add_scheduler(DummyLR)
        self.assertEqual(len(lr_components._addons.components), 1)
        cs = SchedulerChoice().get_hyperparameter_search_space()
        self.assertIn('DummyLR', str(cs))


class OptimizerTest(unittest.TestCase):
    def test_every_optimizer_is_valid(self):
        """
        Makes sure that every optimizer is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        optimizer_choice = OptimizerChoice()

        # Make sure all components are returned
        self.assertEqual(len(optimizer_choice.get_components().keys()), 4)

        # For every optimizer in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, optimizer in optimizer_choice.get_components().items():
            config = optimizer.get_hyperparameter_search_space().sample_configuration()
            estimator = optimizer(**config)
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

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the optimizer
        choice"""
        optimizer_choice = OptimizerChoice()
        cs = optimizer_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(optimizer_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            optimizer_choice.set_hyperparameters(config)

            self.assertEqual(optimizer_choice.choice.__class__,
                             optimizer_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(optimizer_choice.choice))
                self.assertEqual(value, optimizer_choice.choice.__dict__[key])

    def test_optimizer_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        self.assertEqual(len(optimizer_components._addons.components), 0)

        # Then make sure the optimizer can be added and query'ed
        optimizer_components.add_optimizer(DummyOptimizer)
        self.assertEqual(len(optimizer_components._addons.components), 1)
        cs = OptimizerChoice().get_hyperparameter_search_space()
        self.assertIn('DummyOptimizer', str(cs))


class NetworkTest(unittest.TestCase):
    def test_every_network_is_valid(self):
        """
        Makes sure that every network is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        network_choice = NetworkChoice()

        # Make sure all components are returned
        self.assertEqual(len(network_choice.get_components().keys()), 1)

        # For every network in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, network in network_choice.get_components().items():
            config = network.get_hyperparameter_search_space().sample_configuration()
            estimator = network(**config)
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

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the network
        choice"""
        network_choice = NetworkChoice()
        cs = network_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(network_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            network_choice.set_hyperparameters(config)

            self.assertEqual(network_choice.choice.__class__,
                             network_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                # In the case of MLP, parameters are dynamic, so they exist in config
                print(f"vars={vars(network_choice.choice)}")
                parameters = vars(network_choice.choice)
                parameters.update(vars(network_choice.choice)['config'])
                self.assertIn(key, parameters)
                self.assertEqual(value, parameters[key])

    def test_network_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        self.assertEqual(len(network_components._addons.components), 0)

        # Then make sure the scheduler can be added and query'ed
        network_components.add_network(DummyNet)
        self.assertEqual(len(network_components._addons.components), 1)
        cs = NetworkChoice().get_hyperparameter_search_space()
        self.assertIn('DummyNet', str(cs))

    def test_mlp_network_builder(self):
        """Makes sure that we honor the given network architecture
        when building an MLP"""

        X = {
            'num_features': 10,
            'num_classes': 2,
        }
        for num_layers, activation, use_dropout, dictionary in [
            (
                3, 'relu', True, {
                'num_units_1': 11,
                'num_units_2': 18,
                'num_units_3': 11,
                'dropout_1': 0.5,
                'dropout_2': 0.5,
                'dropout_3': 0.5,
                }
            ),
            (
                3, 'relu', False, {
                'num_units_1': 12,
                'num_units_2': 14,
                'num_units_3': 14,
                }
            ),
            (
                5, 'tanh', False, {
                'num_units_1': 12,
                'num_units_2': 14,
                'num_units_3': 14,
                'num_units_4': 17,
                'num_units_5': 14,
                }
            )
        ]:
            network = MLPNet(
                num_layers=num_layers,
                intermediate_activation=activation,
                use_dropout=use_dropout,
                **dictionary,
            )

            # Fit the network and check it's contents
            network.fit(X, y=None)

            # Make sure we properly fitted a module
            self.assertIsInstance(network.network, nn.Sequential)

            # Make sure that every parameter comply with the desired output

            # The last layer has size equal to the number of classes
            self.assertEqual(
                list(network.network.named_modules())[1][1].in_features,
                X['num_features']
            )

            # The last layer has size equal to the number of classes
            self.assertEqual(
                list(network.network.named_modules())[-1][1].out_features,
                X['num_classes']
            )

            # Make sure the number of layers is honored
            layers = [module for name, module in list(network.network.named_modules())
                      if isinstance(module, nn.Linear)]
            self.assertEqual(len(layers), num_layers + 1)

            # Make sure the number of units is honored
            num_units = [module.out_features for name, module in list(
                network.network.named_modules()) if isinstance(module, nn.Linear)]
            self.assertEqual([dictionary['num_units_' + str(i)] for i in range(1, num_layers + 1)
                              ] + [X['num_classes']],
                             num_units
                             )

            dropouts = [module for name, module in list(network.network.named_modules())
                        if isinstance(module, nn.Dropout)]

            if use_dropout:
                self.assertEqual(len(dropouts), num_layers)
            else:
                self.assertEqual(len(dropouts), 0)

            if 'relu' in activation:
                activations = [module for name, module in list(network.network.named_modules())
                               if isinstance(module, nn.ReLU)]
            elif 'tanh' in activation:
                activations = [module for name, module in list(network.network.named_modules())
                               if isinstance(module, nn.Tanh)]
            self.assertEqual(len(activations), num_layers)


if __name__ == '__main__':
    unittest.main()
