# mypy: ignore-errors
import typing
from abc import abstractmethod
from typing import Any, Dict, Optional

from ConfigurationSpace.configuration_space import \
    Configuration, \
    ConfigurationSpace

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.pipeline.base_pipeline import BasePipeline


class BaseTask():
    """
    Base class for the tasks that serve as API to the pipeline and optimizer.
    """

    def __init__(
        self
    ):
        self.pipeline = self.build_pipeline()

        self.pipeline_config = self.pipeline.get_default_pipeline_options()

        self.search_space = self.pipeline.get_hyperparameter_search_space()

    @abstractmethod
    def build_pipeline(self) -> BasePipeline:
        """
        Build the task specific pipeline.
        """
        raise NotImplementedError

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Returns the current pipeline configuration.
        """
        return self.pipeline_config

    def set_pipeline_config(
            self,
            **pipeline_config_kwargs: Any) -> None:
        """
        Check wether arguments are valid and then update the pipeline configuration.
        """
        unknown_keys = []
        for option, value in pipeline_config_kwargs.items:
            if option in self.pipeline_config.keys():
                pass
            else:
                unknown_keys.append(option)

        if len(unknown_keys) > 0:
            raise ValueError("Invalid configuration arguments given %s" % unknown_keys)

        self.pipeline_config.update(pipeline_config_kwargs)

    def get_search_space(self) -> ConfigurationSpace:
        """
        Returns the current search space as ConfigurationSpace object.
        """
        return self.search_space

    def set_search_space(self, search_space: ConfigurationSpace) -> None:
        """
        Update the search space.
        """
        raise NotImplementedError

    @typing.no_type_check
    def search(
        self,
        dataset: BaseDataset,
    ):
        """Start searching for the optimal pipeline.
        Args:
            dataset: (Dataset)
                Auto-PyTorch dataset that provides dataset properties as well as the data
                itself and the training and validation splits.
        """
        # TODO: Check dataset type against task type
        dataset_properties = dataset.get_dataset_properties()

        X = {}
        X.update(dataset_properties)
        X.update(self.pipeline_config)

        self.fit_result = self.pipeline.fit_pipeline(X)

        # TODO do something with the fit result

    @typing.no_type_check
    def fit(
        self,
        dataset: BaseDataset,
        model_config: Configuration,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ):
        """Refit a model configuration and calculate the model performance.
        Given a model configuration, the model is trained on the joint train
        and validation sets of the dataset. This corresponds to the refit
        phase after finding a best hyperparameter configuration after the hpo
        phase.
        Args:
            dataset: (Dataset)
                Auto-PyTorch dataset that provides dataset properties as well as the data
                itself and the training and validation splits.
            model_config: (Configuration)
                The pipeline to fit.
        Returns:
            Value of the evaluation metric calculated on the test set.
        """
        # TODO the model configuration is for the pipeline
        # instead of it being given from the hpo algorithm
        # it takes it from us ?

        dataset_properties = dataset.get_dataset_properties()

        X = {}
        X.update(dataset_properties)
        X.update(self.pipeline_config)

        self.pipeline.set_hyperparameters(model_config)
        self.pipeline.fit(X)

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
        return self.pipeline.predict(X_test)

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
        return self.pipeline.score(X_test, y_test, sample_weights)

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
