from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchPreprocessingAlgorithm(autoPyTorchComponent):
    '''
    Base class for preprocessing algorithms.  
    '''
    def __init__(self) -> None:
        self.preprocessor: BaseEstimator = None

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def get_preprocessor(self) -> BaseEstimator:
        return self.preprocessor
