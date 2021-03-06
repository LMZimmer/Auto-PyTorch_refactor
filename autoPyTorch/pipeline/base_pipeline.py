import warnings
from abc import ABCMeta
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state

import torch

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.create_searchspace_util import (
    add_forbidden,
    find_active_choices,
    get_match_array
)
from autoPyTorch.utils.common import FitRequirement


class BasePipeline(Pipeline):
    """Base class for all pipeline objects.
    Notes
    -----
    This class should not be instantiated, only subclassed.

    Args:
        config (Optional[Configuration]): Allows to directly specify a configuration space
        steps (Optional[List[Tuple[str, autoPyTorchChoice]]]): the list of steps that
            build the pipeline. If provided, they won't be dynamically produced.
        include (Optional[Dict[str, Any]]): Allows the caller to specify which configurations
            to honor during the creation of the configuration space.
        exclude (Optional[Dict[str, Any]]): Allows the caller to specify which configurations
            to avoid during the creation of the configuration space.
        random_state (np.random.RandomState): allows to produce reproducible results by
            setting a seed for randomized settings
        init_params (Optional[Dict[str, Any]])


    Attributes:
        steps (List[Tuple[str, autoPyTorchChoice]]]): the steps of the current pipeline
        config (Configuration): a configuration to delimit the current component choice
        random_state (Optional[np.random.RandomState]): allows to produce reproducible
               results by setting a seed for randomized settings

    """
    __metaclass__ = ABCMeta

    def __init__(
        self,
        config: Optional[Configuration] = None,
        steps: Optional[List[Tuple[str, autoPyTorchChoice]]] = None,
        dataset_properties: Optional[Dict[str, Any]] = None,
        include: Optional[Dict[str, Any]] = None,
        exclude: Optional[Dict[str, Any]] = None,
        random_state: Optional[np.random.RandomState] = None,
        init_params: Optional[Dict[str, Any]] = None
    ):

        self.init_params = init_params if init_params is not None else {}
        self.dataset_properties = dataset_properties if \
            dataset_properties is not None else {}
        self.include = include if include is not None else {}
        self.exclude = exclude if exclude is not None else {}

        if steps is None:
            self.steps = self._get_pipeline_steps(dataset_properties)
        else:
            self.steps = steps

        self.config_space = self.get_hyperparameter_search_space()

        if config is None:
            self.config = self.config_space.get_default_configuration()
        else:
            if isinstance(config, dict):
                config = Configuration(self.config_space, config)
            if self.config_space != config.configuration_space:
                warnings.warn(self.config_space._children)
                warnings.warn(config.configuration_space._children)
                import difflib
                diff = difflib.unified_diff(
                    str(self.config_space).splitlines(),
                    str(config.configuration_space).splitlines())
                diff_msg = '\n'.join(diff)
                raise ValueError('Configuration passed does not come from the '
                                 'same configuration space. Differences are: '
                                 '%s' % diff_msg)
            self.config = config

        self.set_hyperparameters(self.config, init_params=init_params)

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)
        super().__init__(steps=self.steps)

        self._additional_run_info = {}  # type: Dict[str, str]

    def get_max_iter(self) -> int:
        if self.estimator_supports_iterative_fit():
            return self._final_estimator.get_max_iter()
        else:
            raise NotImplementedError()

    def configuration_fully_fitted(self) -> bool:
        return self._final_estimator.configuration_fully_fitted()

    def get_current_iter(self) -> int:
        return self._final_estimator.get_current_iter()

    def predict(self, X: np.ndarray, batch_size: Optional[int] = None
                ) -> np.ndarray:
        """Predict the classes using the selected model.

        Args:
            X (np.ndarray): input data to the array
            batch_size (Optional[int]): batch_size controls whether the pipeline will be
                called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.

        Returns:
            np.ndarray: the predicted values given input X
        """

        # Pre-process X
        if batch_size is None:
            warnings.warn("Batch size not provided. "
                          "Will predict on the whole data in a single iteration")
            batch_size = X.shape[0]
        loader = self.named_steps['data_loader'].get_loader(X=X, batch_size=batch_size)
        return self.named_steps['network'].predict(loader)

    def set_hyperparameters(
        self,
        configuration: Configuration,
        init_params: Optional[Dict] = None
    ) -> 'Pipeline':
        """Method to set the hyperparameter configuration of the pipeline.

        It iterates over the components of the pipeline and applies a given
        configuration accordingly.

        Args:
            configuration (Configuration): configuration object to search and overwrite in
                the pertinent spaces
            init_params (Optional[Dict]): optional initial settings for the config

        """
        self.configuration = configuration

        for node_idx, n_ in enumerate(self.steps):
            node_name, node = n_

            sub_configuration_space = node.get_hyperparameter_search_space(self.dataset_properties)
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('%s:' % node_name):
                    value = configuration[param]
                    new_name = param.replace('%s:' % node_name, '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space,
                                              values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % node_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % node_name, '', 1)
                        sub_init_params_dict[new_name] = value

            if isinstance(node, (autoPyTorchChoice, autoPyTorchComponent, BasePipeline)):
                node.set_hyperparameters(
                    configuration=sub_configuration,
                    init_params=None if init_params is None else sub_init_params_dict,
                )
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self) -> ConfigurationSpace:
        """Return the configuration space for the CASH problem.

        Returns:
            ConfigurationSpace: The configuration space describing the Pipeline.
        """
        if not hasattr(self, 'config_space') or self.config_space is None:
            self.config_space = self._get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties,
                include=self.include,
                exclude=self.exclude,
            )
        return self.config_space

    def get_model(self) -> torch.nn.Module:
        """
        Returns the fitted model to the user
        """
        return self.named_steps['network'].get_network()

    def _get_hyperparameter_search_space(self,
                                         dataset_properties: Dict[str, Any],
                                         include: Optional[Dict[str, Any]] = None,
                                         exclude: Optional[Dict[str, Any]] = None,
                                         ) -> ConfigurationSpace:
        """Return the configuration space for the CASH problem.
        This method should be called by the method
        get_hyperparameter_search_space of a subclass. After the subclass
        assembles a list of available estimators and preprocessor components,
        _get_hyperparameter_search_space can be called to do the work of
        creating the actual ConfigSpace.configuration_space.ConfigurationSpace object.

        Args:
            include (Dict): Overwrite to include user desired components to the pipeline
            exclude (Dict): Overwrite to exclude user desired components to the pipeline

        Returns:
            Configuration: The configuration space describing the AutoPytorch estimator.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Retrieves a str representation of the current pipeline

        Returns:
            str: A formatted representation of the pipeline stages
                 and components
        """
        string = ''
        string += '_' * 40
        string += "\n\t" + self.__class__.__name__ + "\n"
        string += '_' * 40
        string += "\n"
        for i, (stage_name, component) in enumerate(self.named_steps.items()):
            string += str(i) + "-) " + stage_name + ": "
            string += "\n\t"
            string += str(component.choice) if hasattr(component, 'choice') else str(component)
            string += "\n"
            string += "\n"
        string += '_' * 40
        return string

    def _get_base_search_space(
        self,
        cs: ConfigurationSpace,
        dataset_properties: Dict[str, Any],
        include: Optional[Dict[str, Any]],
        exclude: Optional[Dict[str, Any]],
        pipeline: List[Tuple[str, autoPyTorchChoice]]
    ) -> ConfigurationSpace:
        if include is None:
            if self.include is None:
                include = {}
            else:
                include = self.include

        keys = [pair[0] for pair in pipeline]
        for key in include:
            if key not in keys:
                raise ValueError('Invalid key in include: %s; should be one '
                                 'of %s' % (key, keys))

        if exclude is None:
            if self.exclude is None:
                exclude = {}
            else:
                exclude = self.exclude

        keys = [pair[0] for pair in pipeline]
        for key in exclude:
            if key not in keys:
                raise ValueError('Invalid key in exclude: %s; should be one '
                                 'of %s' % (key, keys))

        matches = get_match_array(
            pipeline, dataset_properties, include=include, exclude=exclude)

        # Now we have only legal combinations at this step of the pipeline
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid pipeline found."

        assert np.sum(matches) <= np.size(matches), \
            "'matches' is not binary; %s <= %d, %s" % \
            (str(np.sum(matches)), np.size(matches), str(matches.shape))

        # Iterate each dimension of the matches array (each step of the
        # pipeline) to see if we can add a hyperparameter for that step
        for node_idx, n_ in enumerate(pipeline):
            node_name, node = n_

            is_choice = isinstance(node, autoPyTorchChoice)

            # if the node isn't a choice we can add it immediately because it
            #  must be active (if it wasn't, np.sum(matches) would be zero
            if not is_choice:
                cs.add_configuration_space(
                    node_name,
                    node.get_hyperparameter_search_space(dataset_properties),
                )
            # If the node is a choice, we have to figure out which of its
            #  choices are actually legal choices
            else:
                choices_list = find_active_choices(
                    matches, node, node_idx,
                    dataset_properties,
                    include.get(node_name),
                    exclude.get(node_name)
                )
                sub_config_space = node.get_hyperparameter_search_space(
                    dataset_properties, include=choices_list)
                cs.add_configuration_space(node_name, sub_config_space)

        # And now add forbidden parameter configurations
        # According to matches
        if np.sum(matches) < np.size(matches):
            cs = add_forbidden(
                conf_space=cs, pipeline=pipeline, matches=matches,
                dataset_properties=dataset_properties, include=include,
                exclude=exclude)

        return cs

    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]]
                            ) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoices.

        Returns:
            List[Tuple[str, autoPyTorchChoices]]: list of steps sequentially exercised
                by the pipeline.
        """
        raise NotImplementedError()

    def get_fit_requirements(self) -> List[FitRequirement]:
        """
        Utility function that goes through all the components in
        the pipeline and gets the fit requirement of that components.
        All the fit requirements are then aggregated into a list
        Returns:
            List[NamedTuple]: List of FitRequirements
        """
        fit_requirements = list()  # List[FitRequirement]
        for name, step in self.steps:
            step_requirements = step.get_fit_requirements()
            if step_requirements:
                fit_requirements.extend(step_requirements)

        # remove duplicates in the list
        fit_requirements = list(set(fit_requirements))
        fit_requirements = [req for req in fit_requirements if (req.user_defined and not req.dataset_property)]
        req_names = [req.name for req in fit_requirements]

        # check wether requirement names are unique
        if len(set(req_names)) != len(fit_requirements):
            name_occurences = Counter(req_names)
            multiple_names = [name for name, num_occ in name_occurences.items() if num_occ > 1]
            multiple_fit_requirements = [req for req in fit_requirements if req.name in multiple_names]
            raise ValueError("Found fit requirements with different values %s" % multiple_fit_requirements)
        return fit_requirements

    def get_dataset_requirements(self) -> List[FitRequirement]:
        """
        Utility function that goes through all the components in
        the pipeline and gets the fit requirement that are expected to be
        computed by the dataset for that components. All the fit requirements
        are then aggregated into a list.
        Returns:
            List[NamedTuple]: List of FitRequirements
        """
        fit_requirements = list()  # type: List[FitRequirement]
        for name, step in self.steps:
            step_requirements = step.get_fit_requirements()
            if step_requirements:
                fit_requirements.extend(step_requirements)

        # remove duplicates in the list
        fit_requirements = list(set(fit_requirements))
        fit_requirements = [req for req in fit_requirements if (req.user_defined and req.dataset_property)]
        return fit_requirements

    def _get_estimator_hyperparameter_name(self) -> str:
        """The name of the current pipeline estimator, for representation purposes"""
        raise NotImplementedError()

    def get_additional_run_info(self) -> Dict:
        """Allows retrieving additional run information from the pipeline.
        Can be overridden by subclasses to return additional information to
        the optimization algorithm.

        Returns:
            Dict: Additional information about the pipeline
        """
        return self._additional_run_info

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return {
            'job_id': '1',
            'device': 'cpu',
            'budget_type': 'epochs',
            'epochs': 20,
            'runtime': 3600,
            'torch_num_threads': 1,
            'early_stopping': 10,
            'use_tensorboard_logger': True,
            'use_pynisher': False,
            'metrics_during_training': True
        }
