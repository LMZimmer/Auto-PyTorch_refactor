from enum import Enum
from typing import Any, List, Optional

import numpy as np

import pandas as pd

from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.cross_validation import (
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators
)


class DataTypes(Enum):
    Canonical = 1
    Float = 2
    String = 3


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

    def __init__(self, X: Any, Y: Any):
        X, self.data_types, self.nan_mask, self.itovs, self.vtois = self.interpret(X)
        Y, _, self.target_nan_mask, self.target_itov, self.target_vtoi = self.interpret(Y, assert_single_column=True)
        super().__init__(train_data=(X, Y), shuffle=True)
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

    def interpret(self, data: Any, assert_single_column: bool = False) -> tuple:
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
