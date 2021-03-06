import importlib
import inspect
import pkgutil
import sys
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from sklearn.base import BaseEstimator

from autoPyTorch.utils.common import FitRequirement


def find_components(
        package: str,
        directory: str,
        base_class: BaseEstimator
) -> Dict[str, BaseEstimator]:
    """Utility to find component on a given directory,
    that inherit from base_class
    Args:
        package (str): The associated package that contains the components
        directory (str): The directory from which to extract the components
        base_class (BaseEstimator): base class to filter out desired components
            that don't inherit from this class
    """
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class ThirdPartyComponents(object):
    """
    This class allow the user to create a new component for any stage of the pipeline.
    Inheriting from the base class of each component does not provide any checks,
    to make sure that the hyperparameter space is properly specified.

    This class ensures the minimum component checking for the configuration
    space to work.

    Args:
        base_class (BaseEstimator) component type desired to be created
    """

    def __init__(self, base_class: BaseEstimator) -> None:
        self.base_class = base_class
        self.components = OrderedDict()  # type: Dict[str, BaseEstimator]

    def add_component(self, obj: BaseEstimator) -> None:
        if inspect.isclass(obj) and self.base_class in obj.__bases__:
            name = obj.__name__
            classifier = obj
        else:
            raise TypeError('add_component works only with a subclass of %s' %
                            str(self.base_class))

        properties = set(classifier.get_properties())
        class_specific_properties = classifier.get_required_properties()
        # TODO: Add desired properties when we define them
        should_be_there = {'shortname', 'name'}
        if class_specific_properties is not None:
            should_be_there = should_be_there.union(class_specific_properties)
        for property in properties:
            if property not in should_be_there:
                raise ValueError('Property %s must not be specified for '
                                 'algorithm %s. Only the following properties '
                                 'can be specified: %s' %
                                 (property, name, str(should_be_there)))
        for property in should_be_there:
            if property not in properties:
                raise ValueError('Property %s not specified for algorithm %s' %
                                 (property, name))

        self.components[name] = classifier


class autoPyTorchComponent(BaseEstimator):

    _required_properties: Optional[List[str]] = None

    def __init__(self) -> None:
        super().__init__()
        self._fit_requirements: List[FitRequirement] = list()

    @classmethod
    def get_required_properties(cls) -> Optional[List[str]]:
        """
        Function to get the properties in the component
        that are required for the properly fitting the pipeline.
        Usually defined in the base class of the component
        Returns:
            List[str]: list of properties autopytorch component must have for proper functioning of the pipeline
        """
        return cls._required_properties

    def get_fit_requirements(self) -> Optional[List[FitRequirement]]:
        """
        Function to get the required keys by the component
        that need to be in the fit dictionary
        Returns:
            List[FitRequirement]: a list containing required keys
                            in a named tuple (name: str, type: object)
        """
        return self._fit_requirements

    def add_fit_requirements(self, requirements: List[FitRequirement]) -> None:
        if self._fit_requirements is not None:
            self._fit_requirements.extend(requirements)
        else:
            self._fit_requirements = requirements

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        """Get the properties of the underlying algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]): Describes the dataset
               to work on
        Returns:
            Dict[str, Any]: Properties of the algorithm
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]): Describes the dataset
               to work on

        Returns:
            ConfigurationSpace: The configuration space of this algorithm.
        """
        raise NotImplementedError()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """The fit function calls the fit function of the underlying
        model and returns `self`.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
            y (Any): Not Used -- to comply with API

        Returns:
            self : returns an instance of self.
        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def set_hyperparameters(self,
                            configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> BaseEstimator:
        """
        Applies a configuration to the given component.
        This method translate a hierarchical configuration key,
        to an actual parameter of the autoPyTorch component.

        Args:
            configuration (Configuration): which configuration to apply to
                the chosen component
            init_params (Optional[Dict[str, any]]): Optional arguments to
                initialize the chosen component

        Returns:
            An instance of self
        """
        params = configuration.get_dictionary()

        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
                                     (param, str(self)))
                setattr(self, param, value)

        return self

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        assert isinstance(X, dict), "The input X to the pipeline must be a dictionary"

        if y is not None:
            warnings.warn("Provided y argument, yet only X is required")

        for requirement in self._fit_requirements:
            if requirement.name not in X.keys():
                raise ValueError("To fit {}, expected fit dictionary to have '{}'"
                                 " but got \n {}".format(self.__class__.__name__, requirement.name, list(X.keys())))
            else:
                TYPE_SUPPORTED = isinstance(X[requirement.name], tuple(requirement.supported_types))
                if not TYPE_SUPPORTED:
                    raise TypeError("Expected {} to be instance of {} got {}"
                                    .format(requirement.name,
                                            requirement.supported_types,
                                            type(X[requirement.name])))

    def __str__(self) -> str:
        """Representation of the current Component"""
        name = self.get_properties()['name']
        return "autoPyTorch.pipeline %s" % name
