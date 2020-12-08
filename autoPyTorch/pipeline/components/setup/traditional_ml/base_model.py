from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

import pandas as pd

import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseModelComponent(autoPyTorchSetupComponent):
    """
    Provide an abstract interface for traditional classification methods
    in Auto-Pytorch
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            device: Optional[torch.device] = None
    ) -> None:
        super(BaseModelComponent, self).__init__()
        self.model = None
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchSetupComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        input_shape = X['X_train'].shape[1:]
        if isinstance(X['y_train'], pd.core.series.Series):
            X['y_train'] = X['y_train'].to_numpy()
        output_shape = X['y_train'].shape

        self.model = self.build_model(input_shape=input_shape,
                                      output_shape=output_shape)

        return self

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> torch.nn.Module:
        """
        This method returns a pytorch model, that is dynamically built using
        a self.config that is model specific, and contains the additional
        configuration hyperparameters to build a domain specific model
        """
        raise NotImplementedError()

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        The transform function updates the model in the X dictionary.
        """
        X.update({'model': self.model})
        return X

    def get_model(self) -> nn.Module:
        """
        Return the underlying model object.
        Returns:
            model : the underlying model object
        """
        assert self.model is not None, "No model was initialized"
        return self.model

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        This common utility makes sure that the input dictionary X,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X, y)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.model.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('model', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
