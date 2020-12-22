from typing import Any, Dict, Optional

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.constants import (
    TABULAR_CLASSIFICATION,
    TASK_TYPES_TO_STRING,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset


class TabularClassificationTask(BaseTask):
    def __init__(
        self,
        seed: int = 1,
        n_jobs: int = 1,
        logging_config: Optional[Dict] = None,
        ensemble_size: int = 50,
        ensemble_nbest: int = 50,
        max_models_on_disc: int = 50,
        temporary_directory: Optional[str] = None,
        output_directory: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
    ):
        super().__init__(
            seed=seed,
            n_jobs=n_jobs,
            logging_config=logging_config,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            max_models_on_disc=max_models_on_disc,
            temporary_directory=temporary_directory,
            output_directory=output_directory,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=delete_output_folder_after_terminate,
        )
        self.task_type = TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION]

    def _get_required_dataset_properties(self, dataset: TabularDataset) -> Dict[str, Any]:
        if not isinstance(dataset, TabularDataset):
            raise ValueError("Dataset is incompatible for the given task,: {}".format(
                type(dataset)
            ))
        return {'task_type': dataset.task_type,
                'output_type': dataset.output_type,
                'issparse': dataset.issparse,
                'numerical_columns': dataset.numerical_columns,
                'categorical_columns': dataset.categorical_columns}
