import copy
import unittest
import unittest.mock

from sklearn.base import clone

from autoPyTorch.pipeline.components.setup.lr_scheduler import SchedulerChoice


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

    @unittest.skip("Fit has not been properly coded")
    def test_fit(self):
        """Make sure that fitting a scheduler creates the object.
        Also, test the get_lr method and that on a step the LR changes
        accordingly"""
        pass

    @unittest.skip("Predict has not been properly coded")
    def test_pipeline_predict(self):
        pass


if __name__ == '__main__':
    unittest.main()
