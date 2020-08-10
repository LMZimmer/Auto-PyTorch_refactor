from .base import BaseEncoder
from .encoders import NoneEncoder, OneHotEncoder, OrdinalEncoder 

import os
from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
import numpy as np
from typing import List, Optional

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import find_components

encoding_directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            encoding_directory,
                            BaseEncoder)
# TODO Add possibility for third party components


class EncoderChoice(autoPyTorchChoice):

    def get_components(self) -> OrderedDict:
        components = OrderedDict()
        components.update(_encoders)
        return components

    def get_hyperparameter_search_space(self, dataset_properties: Optional[dict] = None,
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
                    default = default_
                    break

        preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                    list(
                                                        available_preprocessors.keys()),
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
