import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent, find_components
from autoPyTorch.pipeline.components.preprocessing.encoding.base import BaseEncoder

encoding_directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            encoding_directory,
                            BaseEncoder)
# TODO Add possibility for third party components


class EncoderChoice(autoPyTorchChoice):

    def get_components(cls) -> Dict[str, autoPyTorchComponent]:
        components = OrderedDict()
        components.update(_encoders)
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
            raise ValueError("no encoders found, please add a encoder")

        if default is None:
            defaults = ['OneHotEncoder', 'OrdinalEncoder', 'NoneEncoder']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter('__choice__',
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
        return self.choice.transform(X)
