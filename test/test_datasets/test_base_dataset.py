import sys
import unittest

import numpy as np

import pandas as pd

from autoPyTorch.datasets.base_dataset import BaseDataset


class DataFrameMemoryTest(unittest.TestCase):
    def runTest(self):
        memory_limit_mb = 1000
        test_df_small = pd.DataFrame(data=np.random.random((100, 100)))
        test_df_large = pd.DataFrame(data=np.random.random((10000, 10000)))

        self.assertGreater(sys.getsizeof(test_df_large) / 1000000, memory_limit_mb / 2)
        self.assertEqual(BaseDataset.is_small_dataset([test_df_small], memory_limit_mb), True)
        self.assertEqual(BaseDataset.is_small_dataset([test_df_large], memory_limit_mb), False)


class NumpyArrayMemoryTest(unittest.TestCase):
    def runTest(self):
        memory_limit_mb = 1000
        test_array_small = np.random.random((100, 100))
        test_array_large = np.random.random((10000, 10000))

        self.assertGreater(sys.getsizeof(test_array_large) / 1000000, memory_limit_mb / 2)
        self.assertEqual(BaseDataset.is_small_dataset([test_array_small], memory_limit_mb), True)
        self.assertEqual(BaseDataset.is_small_dataset([test_array_large], memory_limit_mb), False)


if __name__ == '__main__':
    unittest.main()
