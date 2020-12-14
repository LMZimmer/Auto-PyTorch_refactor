import logging.handlers
import time
import warnings
from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

from smac.tae import StatusType

from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    IMAGE_TASKS,
    MULTICLASS,
    REGRESSION_TASKS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS,
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.evaluation.utils import (
    convert_multioutput_multiclass_to_multilabel,
    subsampler
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import (
    calculate_score,
    get_metrics,
)
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger
from autoPyTorch.utils.pipeline import get_dataset_requirements

__all__ = [
    'AbstractEvaluator',
    'fit_and_suppress_warnings'
]


class DummyClassificationPipeline(DummyClassifier):
    def __init__(self, config: Configuration,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 init_params: Optional[Dict] = None
                 ) -> None:
        self.configuration = config
        if config == 1:
            super(DummyClassificationPipeline, self).__init__(strategy="uniform")
        else:
            super(DummyClassificationPipeline, self).__init__(strategy="most_frequent")

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        X_train = subsampler(X['X_train'], X['train_indices'])
        y_train = subsampler(X['y_train'], X['train_indices'])
        return super(DummyClassificationPipeline, self).fit(np.ones((X_train.shape[0], 1)), y_train,
                                                            sample_weight=sample_weight)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int = 1000) -> np.array:
        new_X = np.ones((X.shape[0], 1))
        probas = super(DummyClassificationPipeline, self).predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(
            np.float32)
        return probas

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.array:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyClassificationPipeline, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> None:  # pylint: disable=R0201
        return None

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return {}


class DummyRegressionPipeline(DummyRegressor):
    def __init__(self, config: Configuration,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 init_params: Optional[Dict] = None) -> None:
        self.configuration = config
        if config == 1:
            super(DummyRegressionPipeline, self).__init__(strategy='mean')
        else:
            super(DummyRegressionPipeline, self).__init__(strategy='median')

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        X_train = subsampler(X['X_train'], X['train_indices'])
        y_train = subsampler(X['y_train'], X['train_indices'])
        return super(DummyRegressionPipeline, self).fit(np.ones((X_train.shape[0], 1)), y_train,
                                                        sample_weight=sample_weight)

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.array:
        new_X = np.ones((X.shape[0], 1))
        return super(DummyRegressionPipeline, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> None:  # pylint: disable=R0201
        return None

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return {}


def fit_and_suppress_warnings(logger: PicklableClientLogger, pipeline: BaseEstimator,
                              X: Dict[str, Any], y: Any
                              ) -> BaseEstimator:
    def send_warnings_to_log(message, category, filename, lineno,
                             file=None, line=None) -> None:
        logger.debug('%s:%s: %s:%s',
                     filename, lineno, category.__name__, message)
        return

    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        pipeline.fit(X, y)

    return pipeline


class AbstractEvaluator(object):
    def __init__(self, backend: Backend, queue: Queue, metric: autoPyTorchMetric,
                 configuration: Optional[Configuration] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = True,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: bool = False,
                 init_params: Optional[Dict[str, Any]] = None,
                 budget: Optional[float] = None,
                 budget_type: Optional[str] = None,
                 logger_port: Optional[int] = None,
                 all_supported_metrics: bool = True) -> None:

        self.starttime = time.time()

        self.configuration = configuration
        self.backend = backend
        self.queue = queue

        self.datamanager: BaseDataset = self.backend.load_datamanager()
        self.include = include
        self.exclude = exclude

        self.X_train, self.y_train = self.datamanager.train_tensors

        if self.datamanager.val_tensors is not None:
            self.X_valid, self.y_valid = self.datamanager.val_tensors
        else:
            self.X_valid, self.y_valid = None, None

        if self.datamanager.test_tensors is not None:
            self.X_test, self.y_test = self.datamanager.test_tensors
        else:
            self.X_test, self.y_test = None, None

        self.metric = metric

        self.task_type = STRING_TO_TASK_TYPES[self.datamanager.task_type]
        self.output_type = STRING_TO_OUTPUT_TYPES[self.datamanager.output_type]
        self.issparse = self.datamanager.issparse

        self.seed = seed

        self.output_y_hat_optimization = output_y_hat_optimization

        if isinstance(disable_file_output, (bool, list)):
            self.disable_file_output = disable_file_output
        else:
            raise ValueError('disable_file_output should be either a bool or a list')

        self.pipeline_class: Optional[BaseEstimator] = None
        info = {'task_type': self.datamanager.task_type,
                'output_type': self.datamanager.output_type,
                'issparse': self.issparse}
        if self.task_type in REGRESSION_TASKS:
            if not isinstance(self.configuration, Configuration):
                self.pipeline_class = DummyRegressionPipeline
            else:
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = TabularRegressionPipeline
                else:
                    raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_regression
        else:
            if not isinstance(self.configuration, Configuration):
                self.pipeline_class = DummyClassificationPipeline
            else:
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = TabularClassificationPipeline
                elif self.task_type in IMAGE_TASKS:
                    self.pipeline_class = ImageClassificationPipeline
                else:
                    raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_regression
        if self.task_type in TABULAR_TASKS:
            info.update({'numerical_columns': self.datamanager.numerical_columns,
                         'categorical_columns': self.datamanager.categorical_columns})
        self.dataset_properties = self.datamanager.get_dataset_properties(get_dataset_requirements(info))

        self.additional_metrics: Optional[List[autoPyTorchMetric]] = None
        if all_supported_metrics:
            self.additional_metrics = get_metrics(dataset_properties=self.dataset_properties,
                                                  all_supported_metrics=all_supported_metrics)

        self.fit_dictionary = {'dataset_properties': self.dataset_properties}
        self._init_params = init_params
        self.fit_dictionary.update({
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'backend': self.backend,
            'logger_port': logger_port
        })

        if num_run is None:
            num_run = 0
        self.num_run = num_run

        logger_name = '%s(%d)' % (self.__class__.__name__.split('.')[-1],
                                  self.seed)  # TODO: Add name to dataset class
        if logger_port is None:
            logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        self.logger = get_named_client_logger(output_dir=self.backend.temporary_directory, name=logger_name,
                                              port=logger_port)
        self.logger.debug("Dataset Properties created in train_evaluator: {}".format(self.dataset_properties))
        self.Y_optimization = None
        self.Y_actual_train = None

        self.budget = budget
        self.budget_type = budget_type
        default_pipeline_options = self.pipeline_class.get_default_pipeline_options()
        if self.budget_type is not None:
            if self.budget_type == 'runtime':
                default_pipeline_options['runtime'] = self.budget
                if 'epochs' in default_pipeline_options:
                    del default_pipeline_options['epochs']
            elif self.budget_type == 'epochs':
                default_pipeline_options['epochs'] = self.budget
                if 'runtime' in default_pipeline_options:
                    del default_pipeline_options['runtime']
            default_pipeline_options['budget_type'] = self.budget_type
        self.logger.debug("Default pipeline options: {}".format(default_pipeline_options))
        self.fit_dictionary = {**default_pipeline_options, **self.fit_dictionary}

    def _get_pipeline(self) -> BaseEstimator:
        assert self.pipeline_class is not None, "Can't return pipeline, pipeline_class not initialised"
        if not isinstance(self.configuration, Configuration):
            pipeline = self.pipeline_class(config=self.configuration,
                                           random_state=np.random.RandomState(self.seed),
                                           init_params=self.fit_dictionary)
        else:
            pipeline = self.pipeline_class(config=self.configuration,
                                           dataset_properties=self.dataset_properties,
                                           random_state=np.random.RandomState(self.seed),
                                           include=self.include,
                                           exclude=self.exclude,
                                           init_params=self._init_params)
        return pipeline

    def _loss(self, y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
        """SMAC follows a minimization goal, so the make_scorer
        sign is used as a guide to obtain the value to reduce.

        On this regard, to optimize a metric:
            1- score is calculared with calculate_score, with the caveat, that if
            for the metric greater is not better, a negative score is returned.
            2- the err (the optimization goal) is then:
                optimum - (metric.sign * actual_score)
                For accuracy for example: optimum(1) - (+1 * actual score)
                For logloss for example: optimum(0) - (-1 * actual score)
        """

        if not isinstance(self.configuration, Configuration):
            return {self.metric.name: 1.0}

        if self.additional_metrics is not None:
            metrics = self.additional_metrics
        else:
            metrics = [self.metric]
        score = calculate_score(
            y_true, y_hat, self.task_type, metrics)

        err = {metric.name: metric._optimum - score[metric.name] for metric in metrics
               if metric.name in score.keys()}

        return err

    def finish_up(self, loss: Dict[str, float], train_loss: Dict[str, float],
                  opt_pred: np.ndarray, valid_pred: Optional[np.ndarray],
                  test_pred: Optional[np.ndarray], additional_run_info: Optional[Dict],
                  file_output: bool, status: StatusType
                  ) -> Optional[Tuple[float, float, int, Dict]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                opt_pred, valid_pred, test_pred,
            )
        else:
            loss_ = None
            additional_run_info_ = {}

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred
        )

        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        if isinstance(loss, dict):
            loss_ = loss
            loss = loss_[self.metric.name]
        else:
            loss_ = {}

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        for metric_name, value in loss_.items():
            additional_run_info[metric_name] = value
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss

        rval_dict = {'loss': loss,
                     'additional_run_info': additional_run_info,
                     'status': status}

        self.queue.put(rval_dict)
        return None

    def calculate_auxiliary_losses(
            self,
            Y_valid_pred: np.ndarray,
            Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:

        if Y_valid_pred is not None:
            if self.y_valid is not None:
                validation_loss = self._loss(self.y_valid, Y_valid_pred)
                if isinstance(validation_loss, dict):
                    validation_loss = validation_loss[self.metric.name]
            else:
                validation_loss = None
        else:
            validation_loss = None

        if Y_test_pred is not None:
            if self.y_test is not None:
                test_loss = self._loss(self.y_test, Y_test_pred)
                if isinstance(test_loss, dict):
                    test_loss = test_loss[self.metric.name]
            else:
                test_loss = None
        else:
            test_loss = None

        return validation_loss, test_loss

    def file_output(
            self,
            Y_optimization_pred: np.ndarray,
            Y_valid_pred: np.ndarray,
            Y_test_pred: np.ndarray
    ) -> Tuple[Optional[float], Dict]:
        # Abort if self.Y_optimization is None
        # self.Y_optimization can be None if we use partial-cv, then,
        # obviously no output should be saved.
        if self.Y_optimization is None:
            return None, {}

        # Abort in case of shape misalignment
        if self.Y_optimization.shape[0] != Y_optimization_pred.shape[0]:
            return (
                1.0,
                {
                    'error':
                        "Targets %s and prediction %s don't have "
                        "the same length. Probably training didn't "
                        "finish" % (self.Y_optimization.shape, Y_optimization_pred.shape)
                },
            )

        # Abort if predictions contain NaNs
        for y, s in [
            # Y_train_pred deleted here. Fix unittest accordingly.
            [Y_optimization_pred, 'optimization'],
            [Y_valid_pred, 'validation'],
            [Y_test_pred, 'test']
        ]:
            if y is not None and not np.all(np.isfinite(y)):
                return (
                    1.0,
                    {
                        'error':
                            'Model predictions for %s set contains NaNs.' % s
                    },
                )

        # Abort if we don't want to output anything.
        # Since disable_file_output can also be a list, we have to explicitly
        # compare it with True.
        if self.disable_file_output is True:
            return None, {}

        # Notice that disable_file_output==False and disable_file_output==[]
        # means the same thing here.
        if self.disable_file_output is False:
            self.disable_file_output = []

        # This file can be written independently of the others down bellow
        if ('y_optimization' not in self.disable_file_output):
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        if hasattr(self, 'pipelines') and len(self.pipelines) > 0 and self.pipelines[0] is not None:
            if ('pipelines' not in self.disable_file_output):

                if self.task_type in CLASSIFICATION_TASKS:
                    pipelines = VotingClassifier(estimators=None, voting='soft', )
                else:
                    pipelines = VotingRegressor(estimators=None)
                pipelines.estimators_ = self.pipelines
            else:
                pipelines = None
        else:
            pipelines = None

        if hasattr(self, 'pipeline'):
            if 'pipeline' not in self.disable_file_output:
                pipeline = self.pipeline
            else:
                pipeline = None
        else:
            pipeline = None

        self.backend.save_numrun_to_dir(
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            model=pipeline,
            cv_model=pipelines,
            ensemble_predictions=(
                Y_optimization_pred if 'y_optimization' not in self.disable_file_output else None
            ),
            valid_predictions=(
                Y_valid_pred if 'y_valid' not in self.disable_file_output else None
            ),
            test_predictions=(
                Y_test_pred if 'y_test' not in self.disable_file_output else None
            ),
        )

        return None, {}

    def _predict_proba(self, X: np.ndarray, pipeline: BaseEstimator, Y_train: np.ndarray) -> np.ndarray:
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None, line=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = pipeline.predict_proba(X, batch_size=1000)

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def _predict_regression(self, X: np.ndarray, pipeline: BaseEstimator,
                            Y_train: Optional[np.ndarray] = None) -> np.ndarray:
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None, line=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = pipeline.predict(X, batch_size=1000)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction: np.ndarray,
                                       Y_train: np.ndarray) -> np.ndarray:
        assert self.datamanager.num_classes is not None, "Called function on wrong task"
        num_classes: int = self.datamanager.num_classes

        if self.output_type == MULTICLASS and \
                prediction.shape[1] < num_classes:
            if Y_train is None:
                raise ValueError('Y_train must not be None!')
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], num_classes),
                                       dtype=np.float32)

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
