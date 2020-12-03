from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, Subset

from autoPyTorch.datasets.cross_validation import CROSS_VAL_FN, HOLDOUT_FN, is_stratified
from autoPyTorch import constants

BASE_DATASET_INPUT = Union[Tuple[Any, ...], Dataset]


def check_valid_data(data: Any) -> None:
    if not (hasattr(data, '__getitem__') and hasattr(data, '__len__')):
        raise ValueError(
            'The specified Data for Dataset does either not have a __getitem__ or a __len__ attribute.')


def type_check(train_data: BASE_DATASET_INPUT, val_data: Optional[BASE_DATASET_INPUT] = None) -> None:
    if isinstance(train_data, Dataset):
        return
    for i in range(len(train_data)):
        check_valid_data(train_data[i])
    if val_data is not None:
        if isinstance(val_data, Dataset):
            return
        for i in range(len(val_data)):
            check_valid_data(val_data[i])


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 train_data: BASE_DATASET_INPUT,
                 val_data: Optional[BASE_DATASET_INPUT] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42):
        """
        :param train_data: A tuple of objects that have a __len__ and a __getitem__ attribute.
        :param val_data: A optional tuple of objects that have a __len__ and a __getitem__ attribute.
        :param shuffle: Whether to shuffle the data before performing splits
        """
        type_check(train_data, val_data)
        self.train_data = train_data
        self.val_data = val_data
        self.cross_validators: Dict[str, CROSS_VAL_FN] = {}
        self.holdout_validators: Dict[str, HOLDOUT_FN] = {}
        self.rand = np.random.RandomState(seed=seed)
        self.shuffle = shuffle

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        if isinstance(self.train_data, Dataset):
            return self.train_data[index]
        else:
            return tuple(data[index] for data in self.train_data)

    def __len__(self) -> int:
        if isinstance(self.train_data, Dataset):
            return len(self.train_data)
        else:
            return len(self.train_data[0])

    def _get_indices(self) -> np.ndarray:
        if self.shuffle:
            indices = self.rand.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    def _get_targets(self) -> np.ndarray:
        if isinstance(self.train_data, Dataset):
            targets = [self.train_data[idx][-1].numpy() for idx in range(len(self.train_data))]
            targets = np.stack(targets)
        else:
            targets = self.train_data[-1]
        return np.array(targets)

    @abstractmethod
    def get_dataset_properties(self) -> Dict[str, Any]:
        pass

    def create_cross_val_splits(self,
                                cross_val_type: str,
                                num_splits: int) -> List[Tuple[Dataset, Dataset]]:
        if cross_val_type not in self.cross_validators:
            raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not supported.')
        indices = self._get_indices()
        kwargs = {}
        if is_stratified(cross_val_type):
            # we need additional information about the data for stratification
            targets = self._get_targets()
            kwargs["stratify"] = targets[indices]
        splits = self.cross_validators[cross_val_type](num_splits,
                                                       indices,
                                                       **kwargs)
        return [(Subset(self, train_indices), Subset(self, val_indices))
                for train_indices, val_indices in splits]

    def create_val_split(self,
                         holdout_val_type: Optional[str] = None,
                         val_share: Optional[float] = None) -> Tuple[Dataset, Dataset]:
        if val_share is not None:
            if holdout_val_type is None:
                raise ValueError(
                    '`val_share` specified, but `holdout_val_type` not specified.'
                )
            if self.val_data is not None:
                raise ValueError(
                    '`val_share` specified, but the Dataset was a given a pre-defined split at initialization already.')
            if val_share < 0 or val_share > 1:
                raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
            if holdout_val_type not in self.holdout_validators:
                raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')

            indices = self._get_indices()
            kwargs = {}
            if is_stratified(holdout_val_type):
                # we need additional information about the data for stratification
                targets = self._get_targets()
                kwargs["stratify"] = targets[indices]
            train_indices, val_indices = self.holdout_validators[holdout_val_type](val_share,
                                                                                   indices,
                                                                                   **kwargs)
            return Subset(self, train_indices), Subset(self, val_indices)
        else:
            if self.val_data is None:
                raise ValueError('Please specify `val_share` or initialize with a validation dataset.')
            val = BaseDataset(self.val_data)
            return self, val
