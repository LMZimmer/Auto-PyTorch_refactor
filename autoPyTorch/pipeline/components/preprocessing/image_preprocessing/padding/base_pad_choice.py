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
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.padding.base_pad import BasePad


normalise_directory = os.path.split(__file__)[0]
_padders = find_components(__package__,
                           normalise_directory,
                           BasePad)

_addons = ThirdPartyComponents(BasePad)


def add_pad(pad: BasePad) -> None:
    _addons.add_component(pad)


class PadChoice(autoPyTorchChoice):
    """
    Allows for dynamically choosing encoding component at runtime
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available pad components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseNormalise components available
                as choices for encoding the categorical columns
        """
        components = OrderedDict()
        components.update(_padders)
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
            raise ValueError("no image padders found, please add a padder")

        if default is None:
            defaults = ['Pad', 'NoPad']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

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
