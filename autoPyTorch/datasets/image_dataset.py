from abc import ABCMeta
from typing import List, Optional, Tuple, Union, Dict, Any

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset

from torchvision.transforms import functional as TF

from autoPyTorch import constants
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.cross_validation import (
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators
)

IMAGE_DATASET_INPUT = Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]


class _BaseImageDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self,
                 task_type: int,
                 train: IMAGE_DATASET_INPUT,
                 val: Optional[IMAGE_DATASET_INPUT] = None):
        self._task_type = task_type

        _check_image_inputs(train=train, val=val)
        train = _create_image_dataset(data=train)
        if val is not None:
            val = _create_image_dataset(data=val)

        self._mean, self._std = _calc_mean_std(train=train)

        self._dataset_properties = {
            "task_type": self._task_type,
            "mean": self._mean,
            "std": self._std
        }
        super().__init__(train_data=train, val_data=val, shuffle=True)

    def get_dataset_properties(self) -> Dict[str, Any]:
        return self._dataset_properties


class ImageClassificationDataset(_BaseImageDataset):
    def __init__(self,
                 train: IMAGE_DATASET_INPUT,
                 val: Optional[IMAGE_DATASET_INPUT] = None,
                 num_classes: int = None):
        self._num_classes = num_classes

        super().__init__(task_type=constants.IMAGE_CLASSIFICATION, train=train, val=val)

        input_shape, output_shape = self._get_input_output_shape()
        self._dataset_properties.update({
            "input_shape": input_shape,
            "output_shape": output_shape
        })
        self.cross_validators = get_cross_validators(
            CrossValTypes.stratified_k_fold_cross_validation,
            CrossValTypes.k_fold_cross_validation,
            CrossValTypes.shuffle_split_cross_validation,
            CrossValTypes.stratified_shuffle_split_cross_validation
        )

        self.holdout_validators = get_holdout_validators(
            HoldoutValTypes.holdout_validation,
            HoldoutValTypes.stratified_holdout_validation
        )

    def _get_input_output_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        # assume all images are of some shape for now
        # can/must be updated later in the pipeline when image augmentations are performed
        img, _ = self.train_data[0]
        if self._num_classes is not None:
            return tuple(img.shape), (self._num_classes,)
        # determine number of classes by ourselves if it is not given by the user
        targets = self._get_targets()
        num_classes = np.max(targets) + 1
        return tuple(img.shape), (num_classes,)


class ImageRegressionDataset(_BaseImageDataset):
    def __init__(self, train: IMAGE_DATASET_INPUT, val: Optional[IMAGE_DATASET_INPUT] = None):
        super().__init__(task_type=constants.IMAGE_REGRESSION, train=train, val=val)

        input_shape, output_shape = self._get_input_output_shape()
        self._dataset_properties.update({
            "input_shape": input_shape,
            "output_shape": output_shape
        })

        self.cross_validators = get_cross_validators(
            CrossValTypes.k_fold_cross_validation,
            CrossValTypes.shuffle_split_cross_validation
        )

        self.holdout_validators = get_holdout_validators(
            HoldoutValTypes.holdout_validation
        )

    def _get_input_output_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        # assume all images are of same shape for now
        # can/must be updated later in the pipeline when image augmentations are performed
        img, tgt = self.train_data[0]
        return tuple(img.shape), tuple(tgt.shape)


def _calc_mean_std(train: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    C, _, _ = train[0][0].shape

    mean = torch.zeros((C, 1, 1), dtype=torch.float)
    var = torch.zeros((C, 1, 1), dtype=torch.float)

    for i in range(len(train)):
        image = train[i][0].float()

        v, m = torch.var_mean(image, dim=[1, 2], keepdim=True)
        mean += m
        var += v

    mean /= len(train)
    var /= len(train)
    std = torch.sqrt(var)
    return mean, std


def _check_image_inputs(train: IMAGE_DATASET_INPUT,
                        val: Optional[IMAGE_DATASET_INPUT] = None) -> None:
    if not isinstance(train, Dataset):
        assert len(train) == 2, f"expected 2 train inputs, first one being the training data " \
                                f"and second one being the training targets, but got {len(train)} train inputs"
        if len(train[0]) != len(train[1]):
            raise ValueError(
                f"expected train inputs to have the same length, but got lengths {len(train[0])} and {len(train[1])}")
    if val is not None and not isinstance(val, Dataset):
        assert len(val) == 2, f"expected 2 val inputs, first one being the validation data " \
                              f"and second being the validation targets, but got {len(val)} val inputs"
        if len(val[0]) != len(val[1]):
            raise ValueError(
                f"expected val inputs to have the same length, but got lengths {len(train[0])} and {len(train[1])}")


def _create_image_dataset(data: IMAGE_DATASET_INPUT) -> Dataset:
    # if user already provided a dataset, use it
    if isinstance(data, Dataset):
        return data
    # if user provided list of file paths, create a file path dataset
    if isinstance(data[0], list):
        return _FilePathDataset(file_paths=data[0], targets=data[1])
    # if user provided the images as numpy tensors use them directly
    else:
        return TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]))


class _FilePathDataset(Dataset):
    def __init__(self, file_paths: List[str], targets: np.ndarray):
        self.file_paths = file_paths
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(self.file_paths[index], "rb") as f:
            img = Image.open(f).convert("RGB")
        return TF.to_tensor(img), torch.from_numpy(self.targets[index])

    def __len__(self) -> int:
        return len(self.file_paths)
