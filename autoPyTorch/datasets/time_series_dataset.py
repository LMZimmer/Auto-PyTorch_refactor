from abc import ABCMeta
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import torch

from autoPyTorch import constants
from autoPyTorch.datasets.base_dataset import BaseDataset, BASE_DATASET_INPUT
from autoPyTorch.datasets.cross_validation import (
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators
)

# currently only numpy arrays are supported
TIME_SERIES_FORECASTING_INPUT = np.ndarray
TIME_SERIES_REGRESSION_INPUT = Tuple[np.ndarray, np.ndarray]
TIME_SERIES_CLASSIFICATION_INPUT = Tuple[np.ndarray, np.ndarray]


class TimeSeriesForecastingDataset(BaseDataset):
    def __init__(self,
                 target_variables: Tuple[int],
                 sequence_length: int,
                 n_steps: int,
                 train: TIME_SERIES_FORECASTING_INPUT,
                 val: Optional[TIME_SERIES_FORECASTING_INPUT] = None):
        """

        :param target_variables: The indices of the variables you want to forecast
        :param sequence_length: The amount of past data you want to use to forecast future value
        :param n_steps: The number of steps you want to forecast into the future
        :param train: Tuple with one tensor holding the training data
        :param val: Tuple with one tensor holding the validation data
        """
        _check_time_series_forecasting_inputs(
            target_variables=target_variables,
            sequence_length=sequence_length,
            n_steps=n_steps,
            train=train,
            val=val)
        self._mean, self._std = _calc_mean_std(train=train)
        self._min, self._max = _calc_min_max(train=train)
        train = _prepare_time_series_forecasting_tensor(tensor=train,
                                                        target_variables=target_variables,
                                                        sequence_length=sequence_length,
                                                        n_steps=n_steps)
        if val is not None:
            val = _prepare_time_series_forecasting_tensor(tensor=val,
                                                          target_variables=target_variables,
                                                          sequence_length=sequence_length,
                                                          n_steps=n_steps)

        super().__init__(train_data=train, val_data=val, shuffle=False)

        self._input_shape, self._output_shape = self._get_input_output_shape()

        self.cross_validators = get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = get_holdout_validators(HoldoutValTypes.holdout_validation)

    def get_dataset_properties(self) -> Dict[str, Any]:
        return {
            "task_type": constants.TIME_SERIES_FORECASTING,
            "mean": self._mean,
            "std": self._std,
            "min": self._min,
            "max": self._max,
            "input_shape": self._input_shape,
            "output_shape": self._output_shape
        }

    def _get_input_output_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        data, tgt = self.train_data[0]
        return tuple(data.shape), tuple(tgt.shape)


def _check_time_series_forecasting_inputs(target_variables: Tuple[int],
                                          sequence_length: int,
                                          n_steps: int,
                                          train: TIME_SERIES_FORECASTING_INPUT,
                                          val: Optional[TIME_SERIES_FORECASTING_INPUT] = None) -> None:
    if train.ndim != 3:
        raise ValueError(
            "The training data for time series forecasting has to be a three-dimensional tensor of shape PxLxM.")
    if val is not None:
        if val.ndim != 3:
            raise ValueError(
                "The validation data for time series forecasting "
                "has to be a three-dimensional tensor of shape PxLxM.")
    _, time_series_length, num_features = train.shape
    if sequence_length + n_steps > time_series_length:
        raise ValueError(f"Invalid sequence length: Cannot create dataset "
                         f"using sequence_length={sequence_length} and n_steps={n_steps} "
                         f"when the time series are of length {time_series_length}")
    for t in target_variables:
        if t < 0 or t >= num_features:
            raise ValueError(f"Target variable {t} is out of bounds. Number of features is {num_features}, "
                             f"so each target variable has to be between 0 and {num_features - 1}.")


def _prepare_time_series_forecasting_tensor(tensor: np.ndarray,
                                            target_variables: Tuple[int],
                                            sequence_length: int,
                                            n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    population_size, time_series_length, num_features = tensor.shape
    num_targets = len(target_variables)
    num_datapoints = time_series_length - sequence_length - n_steps + 1
    x_tensor = np.zeros((num_datapoints, population_size, sequence_length, num_features), dtype=np.float)
    y_tensor = np.zeros((num_datapoints, population_size, num_targets), dtype=np.float)

    for p in range(population_size):
        for i in range(num_datapoints):
            x_tensor[i, p, :, :] = tensor[p, i:i + sequence_length, :]
            y_tensor[i, p, :] = tensor[p, i + sequence_length + n_steps - 1, target_variables]

    # get rid of population dimension by reshaping
    x_tensor = x_tensor.reshape((-1, sequence_length, num_features))
    y_tensor = y_tensor.reshape((-1, num_targets))
    return x_tensor, y_tensor


class _BaseTimeSeriesDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self,
                 task_type: int,
                 train: BASE_DATASET_INPUT,
                 val: Optional[BASE_DATASET_INPUT] = None):
        self._task_type = task_type

        _check_time_series_inputs(train=train,
                                  val=val,
                                  task_type=self._task_type)

        self._mean, self._std = _calc_mean_std(train=train[0])
        self._min, self._max = _calc_min_max(train=train[0])

        self._dataset_properties = {
            "task_type": self._task_type,
            "mean": self._mean,
            "std": self._std,
            "min": self._min,
            "max": self._max
        }

        super().__init__(train_data=train, val_data=val, shuffle=True)

    def get_dataset_properties(self) -> Dict[str, Any]:
        return self._dataset_properties


class TimeSeriesClassificationDataset(_BaseTimeSeriesDataset):
    def __init__(self,
                 train: TIME_SERIES_CLASSIFICATION_INPUT,
                 val: Optional[TIME_SERIES_CLASSIFICATION_INPUT] = None,
                 num_classes: int = None):
        self._num_classes = num_classes

        super().__init__(task_type=constants.TIME_SERIES_CLASSIFICATION, train=train, val=val)

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
        data, _ = self.train_data[0]
        if self._num_classes is not None:
            return tuple(data.shape), (self._num_classes,)
        # determine number of classes by ourselves if it is not given by the user
        targets = self._get_targets()
        num_classes = np.max(targets) + 1
        return tuple(data.shape), (num_classes,)


class TimeSeriesRegressionDataset(_BaseTimeSeriesDataset):
    def __init__(self, train: Tuple[np.ndarray, np.ndarray], val: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        super().__init__(task_type=constants.TIME_SERIES_REGRESSION, train=train, val=val)

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
        data, tgt = self.train_data[0]
        return tuple(data.shape), tuple(tgt.shape)


def _check_time_series_inputs(task_type: int,
                              train: Union[TIME_SERIES_CLASSIFICATION_INPUT, TIME_SERIES_REGRESSION_INPUT],
                              val: Optional[
                                  Union[TIME_SERIES_CLASSIFICATION_INPUT, TIME_SERIES_REGRESSION_INPUT]] = None
                              ) -> None:
    task_type_str = constants.TASK_TYPES_TO_STRING[task_type]
    if len(train) != 2:
        raise ValueError(f"There must be exactly two training tensors for {task_type_str}. "
                         f"The first one containing the data and the second one containing the targets.")
    if train[0].ndim != 3:
        raise ValueError(
            f"The training data for {task_type_str} has to be a three-dimensional tensor of shape NxSxM.")
    if task_type == constants.TIME_SERIES_CLASSIFICATION:
        if train[1].ndim != 1:
            raise ValueError(
                f"The training targets for {task_type_str} have to be of shape N."
            )
    elif task_type == constants.TIME_SERIES_REGRESSION:
        if train[1].ndim > 2:
            raise ValueError(
                f"The training targets for {task_type_str} have to be of shape N or NxO"
            )
    else:
        raise ValueError(
            f"Invalid task type {task_type_str}"
        )
    if val is not None:
        if len(val) != 2:
            raise ValueError(
                f"There must be exactly two validation tensors for{task_type_str}. "
                f"The first one containing the data and the second one containing the targets.")
        if val[0].ndim != 3:
            raise ValueError(
                f"The validation data for {task_type_str} has to be a "
                f"three-dimensional tensor of shape NxSxM.")
        if task_type == constants.TIME_SERIES_CLASSIFICATION:
            if val[1].ndim != 1:
                raise ValueError(
                    f"The validation targets for {task_type_str} have to be of shape N."
                )
        elif task_type == constants.TIME_SERIES_REGRESSION:
            if val[1].ndim > 2:
                raise ValueError(
                    f"The validation targets for {task_type_str} have to be of shape N or NxO"
                )
        else:
            raise ValueError(
                f"Invalid task type {task_type_str}"
            )


def _calc_mean_std(train: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = np.mean(train, axis=(0, 1), keepdims=True)
    std = np.std(train, axis=(0, 1), keepdims=True)
    return torch.from_numpy(mean.squeeze(0)).float(), torch.from_numpy(std.squeeze(0)).float()


def _calc_min_max(train: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    _min = np.min(train, axis=(0, 1), keepdims=True)
    _max = np.max(train, axis=(0, 1), keepdims=True)
    return torch.from_numpy(_min.squeeze(0)).float(), torch.from_numpy(_max.squeeze(0)).float()
