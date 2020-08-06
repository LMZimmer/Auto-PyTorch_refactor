from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from sklearn.base import BaseEstimator


class autoPyTorchPreprocessingAlgorithm(autoPyTorchComponent):
    def __init__(self):
        self.preprocessor: BaseEstimator = None

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def get_preprocessor(self) -> BaseEstimator:
        return self.preprocessor
