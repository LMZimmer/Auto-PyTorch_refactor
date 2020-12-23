from typing import Any, Dict, List, Optional

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.constants import (
    TABULAR_CLASSIFICATION,
    TASK_TYPES_TO_STRING,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset


class TabularClassificationTask(BaseTask):
    """
    Tabular Classification API to the pipelines.
    Args:
        seed (int): seed to be used for reproducibility.
        n_jobs (int), (default=1): number of consecutive processes to spawn.
        logging_config (Optional[Dict]): specifies configuration
            for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int), (default=50): Number of models added to the ensemble built by
            Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int), (default=50): only consider the ensemble_nbest
            models to build the ensemble
        max_models_on_disc (int), (default=50): maximum number of models saved to disc.
            Also, controls the size of the ensemble as any additional models will be deleted.
            Must be greater than or equal to 1.
        temporary_directory (str): folder to store configuration output and log file
        output_directory (str): folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool): determines whether to delete the temporary directory,
            when finished
        include_components (Optional[List[str]]): If None, all possible components are used.
            Otherwise specifies set of components to use.
        exclude_components (Optional[List[str]]): If None, all possible components are used.
            Otherwise specifies set of components not to use. Incompatible with include
            components
    """
    def __init__(self, seed: int = 1,
                 n_jobs: int = 1,
                 logging_config: Optional[Dict] = None,
                 ensemble_size: int = 1,
                 ensemble_nbest: int = 1, max_models_on_disc: int = 1,
                 temporary_directory: str = './tmp/autoPyTorch_test_tmp',
                 output_directory: str = './tmp/autoPyTorch_test_out',
                 delete_tmp_folder_after_terminate: bool = False,
                 include_components: Optional[List[str]] = None,
                 exclude_components: Optional[List[str]] = None,):
        super().__init__(seed, n_jobs, logging_config, ensemble_size,
                         ensemble_nbest, max_models_on_disc,
                         temporary_directory, output_directory,
                         delete_tmp_folder_after_terminate, include_components,
                         exclude_components)
        self.task_type = TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION]

    def _get_required_dataset_properties(self, dataset) -> Dict[str, Any]:
        assert isinstance(dataset, TabularDataset), "dataset is incompatible for the given task"
        return {'task_type': dataset.task_type,
                'output_type': dataset.output_type,
                'issparse': dataset.issparse,
                'numerical_columns': dataset.numerical_columns,
                'categorical_columns': dataset.categorical_columns}
