# mypy: ignore-errors
import typing
from typing import Optional

from ConfigSpace.configuration_space import \
    Configuration, \
    ConfigurationSpace

import numpy as np

# TODO search_space.saerch_space seems ugly.
# Maybe we can refactor the package or module name
from autoPyTorch.search_space.search_space import SearchSpace


def get_dataset_info(
        X_train: np.ndarray,
        y_train: np.ndarray,
) -> None:
    # placeholder for the function which will
    # be offered by the dataset objects
    pass


def get_pipeline_defaults() -> None:
    # placeholder for the function that will
    # generate default pipeline configs
    pass


class Task():

    def __init__(
        self,
        **pipeline_kwargs,
    ):
        self._pipeline = pipeline_kwargs['pipeline']
        self._pipeline_config = pipeline_kwargs['pipeline_config']
        self._optimizer = pipeline_kwargs['optimizer']
        self._resource_scheduler = ['resource_scheduler']
        self._backend = ['backend']

    @typing.no_type_check
    def search(
        self,
        dataset,
        search_space: SearchSpace,
    ):
        """Refit a model configuration and calculate the model performance.

        Given a model configuration, the model is trained on the joint train
        and validation sets of the dataset. This corresponds to the refit
        phase after finding a best hyperparameter configuration after the hpo
        phase.

        Args:
            dataset: (Dict|Dataset)
                The argument that will provide the dataset splits. It can either
                be a dictionary with the splits, or the dataset object which can
                generate the splits based on different restrictions.
            search_space: (SearchSpace)
                The search space for the hyperparameter optimization algorithm.
        """
        # TODO discuss if we can have the dataset object here for the splits and also for the dataset properties
        # TODO dataset_properties dict ->
        # get defaults for the default pipeline config
        if isinstance(dataset, dict):
            X_train = dataset['X_train']
            y_train = dataset['y_train']
            X_val = dataset['X_val']
            y_val = dataset['y_val']
            X_test = dataset['X_test']
            y_test = dataset['y_test']
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = \
                dataset.get_splits()

        dataset_properties = get_dataset_info(
            X_train,
            y_train,
        )
        pipeline_default_configs = get_pipeline_defaults()

        X = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
        }
        X.update(dataset_properties)
        X.update(pipeline_default_configs)

        """
        if these properties are not in the dataset properties,
        append this dictionary to the X dict
        more_dataset_properties =  {
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'is_small_preprocess': True,
            'categories': [],
        }
        """
        """
        theoretically, this will taken from the pipeline config
        getting the default pipeline config for now. Otherwise, X should be
        updated with this config
        training_config = {
            'job_id': 'example_tabular_classification_1',
            'working_dir': '/tmp/example_tabular_classification_1',  # Hopefully generated by backend
            'device': 'cpu',
            'budget_type': 'epochs',
            'epochs': 100,
            'runtime': 3600,
            'torch_num_threads': 1,
            'early_stopping': 20,
            'use_tensorboard_logger': True,
            'use_pynisher': False,
            'metrics_during_training': True,
        }
        """
        # opt algorithm stuff that will call the pipeline
        self.fit_result = self._pipeline.fit_pipeline(X)
        # TODO do something with the fit result

    @typing.no_type_check
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_config: Configuration,
    ):
        """Refit a model configuration and calculate the model performance.

        Given a model configuration, the model is trained on the joint train
        and validation sets of the dataset. This corresponds to the refit
        phase after finding a best hyperparameter configuration after the hpo
        phase.

        Args:
            X: (np.ndarray)
                The joint train and validation examples of the dataset.
            y: (np.ndarray)
                The joint train and validation labels of the dataset.
            X_test: (np.ndarray)
                The test examples of the dataset.
            y_test: (np.ndarray)
                The test ground truth labels.
            model_config: (Configuration)
                The configuration of the model.
        Returns:
            Value of the evaluation metric calculated on the test set.
        """
        # TODO the model configuration is for the pipeline
        # instead of it being given from the hpo algorithm
        # it takes it from us ?

        # set pipeline hyperparameter configuration
        self._pipeline.set_hyperparameters(
            model_config,
        )
        self._pipeline.fit(X, y)

        return self.score(X_test, y_test)

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Generate the estimator predictions.

        Generate the predictions based on the given examples from the test set.

        Args:
        X_test: (np.ndarray)
            The test set examples.

        Returns:
            Array with estimator predictions.
        """
        # TODO use the batch size and take it from the pipeline
        # TODO argument to method for a flag that might return probabilities
        # in case the pipeline does not have a predict_proba then continue
        # normally and raise warning
        return self._pipeline.predict(X_test)

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the score on the test set.

        Calculate the evaluation measure on the test set.

        Args:
        X_test: (np.ndarray)
            The test examples of the dataset.
        y_test: (np.ndarray)
            The test ground truth labels.
        sample_weights: (np.ndarray|None)
            The weights for each sample.
        Returns:
            Value of the evaluation metric calculated on the test set.
        """
        return self._pipeline.score(X_test, y_test, sample_weights)

    def get_pipeline_config(self) -> Configuration:
        """Get the pipeline configuration.
        Returns:
            A Configuration object which is used to configure the pipeline.
        """
        return self._pipeline_config

    def set_pipeline_config(
        self,
        new_pipeline_config: Configuration,
    ):
        """Sets a new pipeline configuration.

        Args:
        new_pipeline_config (Configuration):
            The new pipeline configuration.
        """
        self._pipeline_config = new_pipeline_config

    @typing.no_type_check
    def get_incumbent_results(
        self
    ):
        pass

    @typing.no_type_check
    def get_incumbent_config(
        self
    ):
        pass

    def get_default_search_space(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ConfigurationSpace:
        """
        Args:
            X_train: (np.ndarray)
                The training examples of the dataset being used.
            y_train: (np.ndarray)
                The training labels of the dataset.
        Returns:
            The config space with the default hyperparameters.
        """
        # get_dataset_info is a placeholder for the real function.
        dataset_properties = get_dataset_info(
            X_train,
            y_train,
        )

        return self._pipeline.get_hyperparameter_search_space(
            dataset_properties,
        )
