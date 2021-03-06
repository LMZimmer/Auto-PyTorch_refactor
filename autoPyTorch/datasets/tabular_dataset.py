from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import pandas as pd

from sklearn.utils import check_array

from autoPyTorch.constants import TABULAR_CLASSIFICATION
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators
)


class DataTypes(Enum):
    Canonical = 1
    Float = 2
    String = 3
    Categorical = 4


class Value2Index(object):
    def __init__(self, values: list):
        assert all(not (pd.isna(v)) for v in values)
        self.values = {v: i for i, v in enumerate(values)}

    def __getitem__(self, item: Any) -> int:
        if pd.isna(item):
            return 0
        else:
            return self.values[item] + 1


class TabularDataset(BaseDataset):
    """
    Support for Numpy Arrays is missing Strings.
    """

    def __init__(self, X: Any, Y: Any,
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None):
        X, self.data_types, self.nan_mask, self.itovs, self.vtois = self.interpret_columns(X)

        if Y is not None:
            Y, _, self.target_nan_mask, self.target_itov, self.target_vtoi = self.interpret_columns(
                Y, assert_single_column=True)
            # For tabular classification, we expect also that it complies with Sklearn
            # The below check_array performs input data checks and make sure that a numpy array
            # is returned, as both Pytorch/Sklearn deal directly with numpy/list objects.
            # In this particular case, the interpret() returns a pandas object (needed to extract)
            # the data types, yet check_array translate the np.array. When Sklearn support pandas
            # the below function will simply return Pandas DataFrame.
            Y = check_array(Y, ensure_2d=False)

        self.categorical_columns, self.numerical_columns, self.categories, self.num_features, self.num_classes = \
            self.infer_dataset_properties(X, Y)

        # Allow support for X_test, Y_test. They will NOT be used for optimization, but
        # rather to have a performance through time on the test data
        if X_test is not None:
            X_test, self._test_data_types, _, _, _ = self.interpret_columns(X_test)

            # Some quality checks on the data
            if self.data_types != self._test_data_types:
                raise ValueError(f"The train data inferred types {self.data_types} are "
                                 "different than the test inferred types {self._test_data_types}")
            if Y_test is not None:
                Y_test, _, _, _, _ = self.interpret_columns(
                    Y_test, assert_single_column=True)
                Y_test = check_array(Y_test, ensure_2d=False)

        super().__init__(train_tensors=(X, Y), test_tensors=(X_test, Y_test), shuffle=True)
        self.task_type = TABULAR_CLASSIFICATION
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

    def interpret_columns(self, data: Any, assert_single_column: bool = False) -> tuple:
        single_column = False
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1 and ',' not in str(data.dtype):
                single_column = True
                data = data[:, None]
            data = pd.DataFrame(data).infer_objects().convert_dtypes()
        elif isinstance(data, pd.DataFrame):
            data = data.infer_objects().convert_dtypes()
        elif isinstance(data, pd.Series):
            single_column = True
            data = data.to_frame()
        else:
            raise ValueError('Provided data needs to be either an np.ndarray or a pd.DataFrame for TabularDataset.')
        if assert_single_column:
            assert single_column, \
                "The data is asserted to be only of a single column, but it isn't. \
                Most likely your targets are not a vector or series."

        data_types = []
        nan_mask = data.isna().to_numpy()
        for col_index, dtype in enumerate(data.dtypes):
            if dtype.kind == 'f':
                data_types.append(DataTypes.Float)
            elif dtype.kind in ('i', 'u', 'b'):
                data_types.append(DataTypes.Canonical)
            elif isinstance(dtype, pd.StringDtype):
                data_types.append(DataTypes.String)
            elif dtype.name == 'category':
                # OpenML format categorical columns as category
                # So add support for that
                data_types.append(DataTypes.Categorical)
            else:
                raise ValueError(f"The dtype in column {col_index} is {dtype} which is not supported.")
        itovs: List[Optional[List[Any]]] = []
        vtois: List[Optional[Value2Index]] = []
        for col_index, (_, col) in enumerate(data.iteritems()):
            if data_types[col_index] != DataTypes.Float:
                non_na_values = [v for v in set(col) if not pd.isna(v)]
                non_na_values.sort()
                itovs.append([np.nan] + non_na_values)
                vtois.append(Value2Index(non_na_values))
            else:
                itovs.append(None)
                vtois.append(None)

        if single_column:
            return data.iloc[:, 0], data_types[0], nan_mask[0], itovs[0], vtois[0]

        return data, data_types, nan_mask, itovs, vtois

    def infer_dataset_properties(self, X: Any, y: Any) \
            -> Tuple[List[int], List[int], List[object], int, Optional[int]]:

        categorical_columns = []
        numerical_columns = []
        for i, data_type in enumerate(self.data_types):
            if data_type == DataTypes.String or data_type == DataTypes.Categorical:
                categorical_columns.append(i)
            else:
                numerical_columns.append(i)
        categories = [np.unique(X.iloc[:, a]).tolist() for a in categorical_columns]
        num_features = X.shape[1]
        num_classes = None
        if y is not None:
            num_classes = len(np.unique(y))

        return categorical_columns, numerical_columns, categories, num_features, num_classes
