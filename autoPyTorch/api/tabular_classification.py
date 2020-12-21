from typing import Any, Dict

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.constants import TASK_TYPES_TO_STRING, TABULAR_CLASSIFICATION
from autoPyTorch.datasets.tabular_dataset import TabularDataset


class TabularClassificationTask(BaseTask):
    def __init__(self, seed: int = 1, n_jobs: int = 1, logging_config=None, ensemble_size: int = 1,
                 ensemble_nbest: int = 1, max_models_on_disc: int = 1,
                 temporary_directory: str = './tmp/autoPyTorch_test_tmp',
                 output_directory: str = './tmp/autoPyTorch_test_out',
                 delete_tmp_folder_after_terminate: bool = False):
        super().__init__(seed, n_jobs, logging_config, ensemble_size, ensemble_nbest, max_models_on_disc,
                         temporary_directory, output_directory, delete_tmp_folder_after_terminate)
        self.task_type = TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION]

    def _get_required_dataset_properties(self, dataset) -> Dict[str, Any]:
        assert isinstance(dataset, TabularDataset), "dataset is incompatible for the given task"
        return {'task_type': dataset.task_type,
                'output_type': dataset.output_type,
                'issparse': dataset.issparse,
                'numerical_columns': dataset.numerical_columns,
                'categorical_columns': dataset.categorical_columns}
