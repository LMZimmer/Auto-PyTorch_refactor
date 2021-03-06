import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder import BaseEncoder


encoding_directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            encoding_directory,
                            BaseEncoder)
_addons = ThirdPartyComponents(BaseEncoder)


def add_encoder(encoder: BaseEncoder) -> None:
    _addons.add_component(encoder)


class EncoderChoice(autoPyTorchChoice):
    """
    Allows for dynamically choosing encoding component at runtime
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available encoder components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseEncoder components available
                as choices for encoding the categorical columns
        """
        components = OrderedDict()
        components.update(_encoders)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(self,
                                        dataset_properties: Optional[Dict[str, Any]] = None,
                                        default: Optional[str] = None,
                                        include: Optional[List[str]] = None,
                                        exclude: Optional[List[str]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = dict()

        dataset_properties = {**self.dataset_properties, **dataset_properties}

        available_preprocessors = self.get_available_components(dataset_properties=dataset_properties,
                                                                include=include,
                                                                exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError("no encoders found, please add a encoder")

        if default is None:
            defaults = ['OneHotEncoder', 'OrdinalEncoder', 'NoEncoder']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        # add only no encoder to choice hyperparameters in case the dataset is only numerical
        if len(dataset_properties['categorical_columns']) == 0:
            default = 'NoEncoder'
            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         ['NoEncoder'],
                                                         default_value=default)
        else:
            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         list(available_preprocessors.keys()),
                                                         default_value=default)

        cs.add_hyperparameter(preprocessor)

        # add only child hyperparameters of early_preprocessor choices
        for name in preprocessor.choices:
            preprocessor_configuration_space = available_preprocessors[name].\
                get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, preprocessor_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _check_dataset_properties(self, dataset_properties: Dict[str, Any]) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.
        Args:
            dataset_properties:

        """
        super()._check_dataset_properties(dataset_properties)
        assert 'numerical_columns' in dataset_properties.keys(), \
            "Dataset properties must contain information about numerical columns"
        assert 'categorical_columns' in dataset_properties.keys(), \
            "Dataset properties must contain information about categorical columns"
