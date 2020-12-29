import os
import shutil
import unittest
import unittest.mock

import numpy as np

import pandas as pd

import pytest

from sklearn.datasets import make_classification

from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import get_preprocess_transforms
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
            'input_shape': (self.num_features,),
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
        self.backend.save_datamanager(datamanager)

        self.fit_dictionary = {
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'numerical_columns': list(range(self.num_features)),
            'categorical_columns': [],
            'categories': [],
            'X_train': self.X,
            'y_train': self.y,
            'train_indices': list(range(self.X.shape[0] // 2)),
            'val_indices': list(range(self.X.shape[0] // 2, self.X.shape[0])),
            'is_small_preprocess': False,
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

    def tearDown(self):
        self.backend.context.delete_directories()


@pytest.mark.parametrize("fit_dictionary", ['fit_dictionary_numerical_only',
                                            'fit_dictionary_categorical_only',
                                            'fit_dictionary_num_and_categorical'], indirect=True)
class TestTabularClassification:
    def test_pipeline_fit(self, fit_dictionary):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""

        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'])
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)
        pipeline.fit(fit_dictionary)

        # To make sure we fitted the model, there should be a
        # run summary object with accuracy
        run_summary = pipeline.named_steps['trainer'].run_summary
        assert run_summary is not None

        # Make sure that performance was properly captured
        assert run_summary.performance_tracker['train_loss'][1] > 0
        assert run_summary.total_parameter_count > 0
        assert 'accuracy' in run_summary.performance_tracker['train_metrics'][1]

    def test_pipeline_predict(self, fit_dictionary):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""
        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'])

        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        X_train = np.copy(fit_dictionary['X_train'])
        pipeline.fit(fit_dictionary)

        prediction = pipeline.predict(
            pd.DataFrame(X_train).infer_objects().convert_dtypes())
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (200, 2)

    def test_pipeline_predict_proba(self, fit_dictionary):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline
        And then predict using predict probability
        """
        if len(fit_dictionary['dataset_properties']['categorical_columns']) <= 0:
            pytest.skip("Numerical only predict probabilities is not yet supported")
        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'])

        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        X_train = np.copy(fit_dictionary['X_train'])
        pipeline.fit(fit_dictionary)

        prediction = pipeline.predict_proba(
            pd.DataFrame(X_train).infer_objects().convert_dtypes())
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (200, 2)

    def test_pipeline_transform(self, fit_dictionary):
        """
        In the context of autopytorch, transform expands a fit dictionary with
        components that where previously fit. We can use this as a nice way to make sure
        that fit properly work.
        This code is added in light of components not properly added to the fit dicitonary
        """

        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'])
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        pipeline.fit(fit_dictionary)

        # We do not want to make the same early preprocessing operation to the fit dictionary
        if 'X_train' in fit_dictionary:
            fit_dictionary.pop('X_train')

        transformed_fit_dictionary = pipeline.transform(fit_dictionary)

        # First, we do not lose anyone! (We use a fancy subset containment check)
        assert fit_dictionary.items() <= transformed_fit_dictionary.items()

        # Then the pipeline should have added the following keys
        expected_keys = {'imputer', 'encoder', 'scaler', 'tabular_transformer',
                         'preprocess_transforms', 'network', 'optimizer', 'lr_scheduler',
                         'train_data_loader', 'val_data_loader', 'run_summary'}
        assert expected_keys.issubset(set(transformed_fit_dictionary.keys()))

        # Then we need to have transformations being created.
        assert len(get_preprocess_transforms(transformed_fit_dictionary)) > 0

        # We expect the transformations to be in the pipeline at anytime for inference
        assert 'preprocess_transforms' in transformed_fit_dictionary.keys()

    @pytest.mark.parametrize("is_small_preprocess", [True, False])
    def test_default_configuration(self, fit_dictionary, is_small_preprocess):
        """Makes sure that when no config is set, we can trust the
        default configuration from the space"""

        fit_dictionary['is_small_preprocess'] = is_small_preprocess

        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'])

        pipeline.fit(fit_dictionary)

    def test_remove_key_check_requirements(self, fit_dictionary):
        """Makes sure that when a key is removed from X, correct error is outputted"""
        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'])
        for key in ['job_id', 'device', 'split_id', 'use_pynisher', 'torch_num_threads',
                    'dataset_properties', ]:
            fit_dictionary_copy = fit_dictionary.copy()
            fit_dictionary_copy.pop(key)
            with pytest.raises(ValueError, match=r"To fit .+?, expected fit dictionary to have"):
                pipeline.fit(fit_dictionary_copy)

    def test_network_optimizer_lr_handshake(self, fit_dictionary):
        """Fitting a network should put the network in the X"""
        # Create the pipeline to check. A random config should be sufficient
        dataset_properties = {
            'numerical_columns': [],
            'categorical_columns': [],
            'task_type': 'tabular_classification',
            'input_shape': (10,),
            'num_classes': 2,
        }
        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'])
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        # Make sure that fitting a network adds a "network" to X
        self.assertIn('network', pipeline.named_steps.keys())
        fit_dictionary = {'dataset_properties': dataset_properties, 'X_train': self.X, 'y_train': self.y}
        X = pipeline.named_steps['network'].search(
            {'dataset_properties': dataset_properties, 'X_train': self.X, 'y_train': self.y},
            None
        ).transform(fit_dictionary)
        assert 'network' in X

        # Then fitting a optimizer should fail if no network:
        self.assertIn('optimizer', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError, r"To fit .+?, expected fit dictionary to have 'network' but got .*"):
            pipeline.named_steps['optimizer'].search({'dataset_properties': {}}, None)

        # No error when network is passed
        X = pipeline.named_steps['optimizer'].search(X, None).transform(X)
        self.assertIn('optimizer', X)

        # Then fitting a optimizer should fail if no network:
        self.assertIn('lr_scheduler', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError,
                                    r"To fit .+?, expected fit dictionary to have 'optimizer' but got .*"):
            pipeline.named_steps['lr_scheduler'].search({'dataset_properties': {}}, None)

        # No error when network is passed
        X = pipeline.named_steps['lr_scheduler'].search(X, None).transform(X)
        self.assertIn('optimizer', X)

    def test_get_fit_requirements(self, fit_dictionary):
        dataset_properties = {'numerical_columns': [], 'categorical_columns': []}
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        fit_requirements = pipeline.get_fit_requirements()

        # check if fit requirements is a list of FitRequirement named tuples
        assert isinstance(fit_requirements, list)
        for requirement in fit_requirements:
            assert isinstance(requirement, FitRequirement)
