import os
import shutil
import unittest
import unittest.mock

from sklearn.datasets import make_classification

from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.backend import create
from autoPyTorch.utils.common import FitRequirement


class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.num_features = 4
        self.num_classes = 2
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=self.num_features,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=self.num_classes,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0
        )
        self.dataset_properties = {
            'task_type': 'tabular_classification',
            'output_type': 'binary',
            'numerical_columns': list(range(4)),
            'categorical_columns': [],
            'categories': [],
            'is_small_preprocess': False,
            'issparse': False,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
        }

        # Create run dir
        tmp_dir = '/tmp/autoPyTorch_ensemble_test_tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        output_dir = '/tmp/autoPyTorch_ensemble_test_out'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        self.backend = create(
            temporary_directory=tmp_dir,
            output_directory=output_dir,
            delete_tmp_folder_after_terminate=False
        )

        # Create the directory structure
        self.backend._make_internals_directory()

        # Create a datamanager for this toy problem
        datamanager = TabularDataset(
            X=self.X, Y=self.y,
            X_test=self.X, Y_test=self.y,
        )
        datamanager.create_splits()
        self.backend.save_datamanager(datamanager)

    def tearDown(self):
        self.backend.context.delete_directories()

    def test_pipeline_fit(self):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""

        pipeline = TabularClassificationPipeline(dataset_properties=self.dataset_properties)
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)
        pipeline.fit({'X_train': self.X,
                      'y_train': self.y,
                      'train_indices': list(range(self.X.shape[0] // 2)),
                      'val_indices': list(range(self.X.shape[0] // 2, self.X.shape[0])),
                      # Training configuration
                      'dataset_properties': self.dataset_properties,
                      'job_id': 'example_tabular_classification_1',
                      'device': 'cpu',
                      'budget_type': 'epochs',
                      'epochs': 5,
                      'torch_num_threads': 1,
                      'early_stopping': 20,
                      'working_dir': '/tmp',
                      'use_tensorboard_logger': True,
                      'use_pynisher': False,
                      'metrics_during_training': True,
                      'split_id': 0,
                      'backend': self.backend,
                      }
                     )

        # To make sure we fitted the model, there should be a
        # run summary object with accuracy
        self.assertIsNotNone(pipeline.named_steps['trainer'].run_summary)

    def test_default_configuration(self):
        """Makes sure that when no config is set, we can trust the
        default configuration from the space"""
        pipeline = TabularClassificationPipeline(dataset_properties=self.dataset_properties)

        pipeline.fit(
            {'X_train': self.X,
             'y_train': self.y,
             'train_indices': list(range(self.X.shape[0] // 2)),
             'val_indices': list(range(self.X.shape[0] // 2, self.X.shape[0])),
             # Training configuration
             'dataset_properties': self.dataset_properties,
             'job_id': 'example_tabular_classification_1',
             'device': 'cpu',
             'budget_type': 'epochs',
             'epochs': 5,
             'torch_num_threads': 1,
             'early_stopping': 20,
             'working_dir': '/tmp',
             'use_tensorboard_logger': True,
             'use_pynisher': False,
             'metrics_during_training': True,
             'split_id': 0,
             'backend': self.backend,
             }
        )

    def test_remove_key_check_requirements(self):
        """Makes sure that when a key is removed from X, correct error is outputted"""
        pipeline = TabularClassificationPipeline(dataset_properties=self.dataset_properties)
        X = {'X_train': self.X,
             'y_train': self.y,
             'train_indices': list(range(self.X.shape[0] // 2)),
             'val_indices': list(range(self.X.shape[0] // 2, self.X.shape[0])),
             # Training configuration
             'dataset_properties': self.dataset_properties,
             'job_id': 'example_tabular_classification_1',
             'device': 'cpu',
             'budget_type': 'epochs',
             'epochs': 5,
             'torch_num_threads': 1,
             'early_stopping': 20,
             'working_dir': '/tmp',
             'use_tensorboard_logger': True,
             'use_pynisher': False,
             'metrics_during_training': True,
             'split_id': 0,
             'backend': self.backend,
             }
        for key in X.keys():
            # skip tests for data loader requirements as data loader has different check_requirements
            if key == 'y_train' or 'val_indices':
                continue
            X_copy = X.copy()
            X_copy.pop(key)
            try:
                pipeline.fit(X_copy)
            except ValueError as msg:
                self.assertRegex(str(msg), r"To fit .+?, expected fit dictionary to have .+? but got .*")

    def test_network_optimizer_lr_handshake(self):
        """Fitting a network should put the network in the X"""

        # Create the pipeline to check. A random config should be sufficient
        dataset_properties = {
            'numerical_columns': [],
            'categorical_columns': [],
            'task_type': 'tabular_classification',
            'num_features': 10,
            'num_classes': 2,
        }
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        # Make sure that fitting a network adds a "network" to X
        self.assertIn('network', pipeline.named_steps.keys())
        fit_dictionary = {'dataset_properties': dataset_properties, 'X_train': self.X, 'y_train': self.y}
        X = pipeline.named_steps['network'].fit(
            {'dataset_properties': dataset_properties, 'X_train': self.X, 'y_train': self.y},
            None
        ).transform(fit_dictionary)
        self.assertIn('network', X)

        # Then fitting a optimizer should fail if no network:
        self.assertIn('optimizer', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError, r"To fit .+?, expected fit dictionary to have 'network' but got .*"):
            pipeline.named_steps['optimizer'].fit({'dataset_properties': {}}, None)

        # No error when network is passed
        X = pipeline.named_steps['optimizer'].fit(X, None).transform(X)
        self.assertIn('optimizer', X)

        # Then fitting a optimizer should fail if no network:
        self.assertIn('lr_scheduler', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError,
                                    r"To fit .+?, expected fit dictionary to have 'optimizer' but got .*"):
            pipeline.named_steps['lr_scheduler'].fit({'dataset_properties': {}}, None)

        # No error when network is passed
        X = pipeline.named_steps['lr_scheduler'].fit(X, None).transform(X)
        self.assertIn('optimizer', X)

    def test_get_fit_requirements(self):
        dataset_properties = {'numerical_columns': [], 'categorical_columns': []}
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        fit_requirements = pipeline.get_fit_requirements()

        # check if fit requirements is a list of FitRequirement named tuples
        self.assertIsInstance(fit_requirements, list)
        for requirement in fit_requirements:
            self.assertIsInstance(requirement, FitRequirement)


if __name__ == '__main__':
    unittest.main()
