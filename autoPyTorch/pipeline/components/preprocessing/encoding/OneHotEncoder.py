from typing import Optional, Union

import numpy as np

from sklearn.preprocessing import OneHotEncoder as OHE

from autoPyTorch.pipeline.components.preprocessing.encoding import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """
    Encode categorical features as a one-hot numerical array
    """
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.random_state = random_state
        self.preprocessor = OHE(categories='auto', sparse=False, handle_unknown='error')
