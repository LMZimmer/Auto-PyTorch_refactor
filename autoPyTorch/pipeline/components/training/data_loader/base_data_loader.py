from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.utils import check_array

import torch

import torchvision


from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.common import FitRequirement, custom_collate_fn


class BaseDataLoaderComponent(autoPyTorchTrainingComponent):
    """This class is an interface to the PyTorch Dataloader.

    It gives the possibility to read various types of mapped
    datasets as described in:
    https://pytorch.org/docs/stable/data.html

    """

    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_data_loader = None  # type: Optional[torch.utils.data.DataLoader]
        self.val_data_loader = None  # type: Optional[torch.utils.data.DataLoader]

        # We also support existing datasets!
        self.dataset = None
        self.vision_datasets = self.get_torchvision_datasets()

        # Save the transformations for reuse
        self.train_transform = None  # type: Optional[torchvision.transforms.Compose]
        self.val_transform = None  # type: Optional[torchvision.transforms.Compose]

        # Define fit requirements
        self.add_fit_requirements([
            FitRequirement("split_id", (int,), user_defined=True, dataset_property=False),
            FitRequirement("train_indices", (List,), user_defined=True, dataset_property=False),
            FitRequirement("Backend", (Backend,), user_defined=True, dataset_property=False),
            FitRequirement("is_small_preprocess", (bool,), user_defined=True, dataset_property=True)])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'train_data_loader': self.train_data_loader,
                  'val_data_loader': self.val_data_loader})
        return X

    def fit(self, X: Dict[str, Any], y: Any = None) -> torch.utils.data.DataLoader:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Make sure there is an optimizer
        self.check_requirements(X, y)

        self.train_transform = self.build_transform(X, train=True)
        self.val_transform = self.build_transform(X, train=False)

        datamanager = X['backend'].load_datamanager()
        if X["is_small_preprocess"]:
            # This parameter indicates that the data has been pre-processed for speed
            # Overwrite the datamanager with the pre-processes data
            datamanager.replace_data(X['X_train'], X['X_test'] if 'X_test' in X else None)
        train_dataset, val_dataset = datamanager.get_dataset_for_training(split_id=X['split_id'])
        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', True),
            collate_fn=custom_collate_fn,
        )

        self.val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', False),
            collate_fn=custom_collate_fn,
        )

        return self

    def get_loader(self, X: np.ndarray, y: Optional[np.ndarray] = None, batch_size: int = np.inf,
                   ) -> torch.utils.data.DataLoader:
        """
        Creates a data loader object from the provided data,
        applying the transformations meant to validation objects
        """

        # We need the proper dtype on the data
        # This has to be changed according to
        X = check_array(X)
        if y:
            y = check_array(y, ensure_2d=False)
        dataset = BaseDataset(
            train_tensors=(X, y),
            transforms=self.val_transform
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

    def build_transform(self, X: Dict[str, Any], train: bool = True) -> torchvision.transforms.Compose:
        """
        Method to build a transformation that can pre-process input data

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            train (bool)" whether transformation to be built are for training of test mode"

        Returns:
            A composition of transformations
        """
        raise NotImplementedError()

    def get_train_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the train data

        Returns:
            torch.utils.data.DataLoader: A train data loader
        """
        assert self.train_data_loader is not None, "No train data loader fitted"
        return self.train_data_loader

    def get_val_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the validation data

        Returns:
            torch.utils.data.DataLoader: A validation data loader
        """
        assert self.val_data_loader is not None, "No val data loader fitted"
        return self.val_data_loader

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

        # make sure the parent requirements are honored
        super().check_requirements(X, y)

        # We allow reading data from a user provided dataset
        # or from X, Y pairs
        if 'split_id' not in X:
            raise ValueError("split_id is needed to select the respampled dataset. "
                             "Currently X={}.".format(
                                 X
                             )
                             )
        if 'backend' not in X:
            raise ValueError("backend is needed to load the data from disk")

        if 'is_small_preprocess' not in X:
            raise ValueError("is_small_pre-process is required to know if the data was preprocessed"
                             " or if the data-loader should transform it while loading a batch")

        # We expect this class to be a base for image/tabular/time
        # And the difference among this data types should be mainly
        # in the transform, so we delegate for special transformation checking
        # to the below method
        self._check_transform_requirements(X, y)

    def _check_transform_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """

        Makes sure that the fit dictionary contains the required transformations
        that the dataset should go through

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        raise NotImplementedError()

    def get_torchvision_datasets(self) -> Dict[str, torchvision.datasets.VisionDataset]:
        """ Returns the supported dataset classes from torchvision

        This is gonna be used to instantiate a dataset object for the dataloader

        Returns:
            Dict[str, torchvision.datasets.VisionDataset]: A mapping from dataset name to class

        """
        return {
            'FashionMNIST': torchvision.datasets.FashionMNIST,
            'MNIST': torchvision.datasets.MNIST,
            'CIFAR10': torchvision.datasets.CIFAR10,
            'CIFAR100': torchvision.datasets.CIFAR100,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None
                                        ) -> ConfigurationSpace:
        batch_size = UniformIntegerHyperparameter(
            "batch_size", 32, 320, default_value=64)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([batch_size])
        return cs

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.train_data_loader.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('train_data_loader', None)
        info.pop('val_data_loader', None)
        info.pop('vision_datasets', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
