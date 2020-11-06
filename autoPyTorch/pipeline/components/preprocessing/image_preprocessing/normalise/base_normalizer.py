from typing import Any, Dict, List, Optional

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.base_image_preprocessor import \
    autoPyTorchImagePreprocessingComponent
from autoPyTorch.utils.common import FitRequirement


class BaseNormalizer(autoPyTorchImagePreprocessingComponent):

    def __init__(self) -> None:
        super(BaseNormalizer, self).__init__()
        self._fit_requirements = [FitRequirement('channelwise_mean', (np.ndarray,)),
                                  FitRequirement('channelwise_std', (np.ndarray,))]
        super_requirements: Optional[List[FitRequirement]] = super().get_fit_requirements()
        if super_requirements:
            self._fit_requirements.extend(super_requirements)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({'normalise': self})
        return X

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
        super().check_requirements(X, y)

        if 0 in X['channelwise_std']:
            raise ZeroDivisionError("Can't normalise when std is zero")

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
