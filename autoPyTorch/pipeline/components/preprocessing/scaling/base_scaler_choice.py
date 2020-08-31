import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.preprocessing.scaling.base_scaler import BaseScaler

scaling_directory = os.path.split(__file__)[0]
_scalers = find_components(__package__,
                           scaling_directory,
                           BaseScaler)

_addons = ThirdPartyComponents(BaseScaler)


def add_scaler(scaler: BaseScaler) -> None:
    _addons.add_component(scaler)


class ScalerChoice(autoPyTorchChoice):
    """
    Allows for dynamically choosing scaling component at runtime
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available scaler components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseScalers components available
                as choices for scaling
        """
        components = OrderedDict()
        components.update(_scalers)
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

        available_preprocessors = self.get_available_components(dataset_properties=dataset_properties,
                                                                include=include,
                                                                exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError("no rescalers found, please add a rescaler")

        if default is None:
            defaults = ['Normalizer', 'StandardScaler', 'MinMaxScaler', 'NoneScaler']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                     list(available_preprocessors.keys()),
                                                     default_value=default)
        cs.add_hyperparameter(preprocessor)

        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[name].\
                get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, preprocessor_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.choice is not None, "Can not call transform without initialising choice"
        return self.choice.transform(X)  # type: ignore
