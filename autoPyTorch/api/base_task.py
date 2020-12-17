from abc import abstractmethod
import copy
import joblib
import json
import multiprocessing
import os
import time
import typing
from typing import Any, Callable, Dict, Optional, List
import tempfile
import warnings

from ConfigSpace.configuration_space import \
    Configuration, \
    ConfigurationSpace

import dask

import numpy as np

import pandas as pd

import sklearn
from sklearn.utils.validation import check_is_fitted
from sklearn.dummy import DummyClassifier, DummyRegressor

from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import CrossValTypes
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.stopwatch import StopWatch
from autoPyTorch.utils.backend import create, Backend
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.utils.logging_ import setup_logger, start_log_server, get_named_client_logger, PicklableClientLogger
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilderManager
from autoPyTorch.ensemble.singlebest_ensemble import SingleBest
from autoPyTorch.constants import STRING_TO_TASK_TYPES, STRING_TO_OUTPUT_TYPES, REGRESSION_TASKS
from autoPyTorch.evaluation.abstract_evaluator import fit_and_suppress_warnings


def _model_predict(model, X, batch_size, logger, task):
    def send_warnings_to_log(
            message, category, filename, lineno, file=None, line=None):
        logger.debug('%s:%s: %s:%s' % (filename, lineno, category.__name__, message))
        return

    X_ = X.copy()
    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        if task in REGRESSION_TASKS:
            prediction = model.predict(X_, batch_size=batch_size)
        else:
            prediction = model.predict_proba(X_, batch_size=batch_size)

            # Check that all probability values lie between 0 and 1.
            assert (
                    (prediction >= 0).all() and (prediction <= 1).all()
            ), "For {}, prediction probability not within [0, 1]!".format(
                model
            )

    if len(prediction.shape) < 1 or len(X_.shape) < 1 or \
            X_.shape[0] < 1 or prediction.shape[0] != X_.shape[0]:
        logger.warning(
            "Prediction shape for model %s is %s while X_.shape is %s",
            model, str(prediction.shape), str(X_.shape)
        )
    return prediction


class BaseTask:
    """
    Base class for the tasks that serve as API to the pipelines.
    """

    def __init__(
            self,
            logging_config=None,
            temporary_directory: str = './tmp/autoPyTorch_smac_test_tmp',
            output_directory: str = './tmp/autoPyTorch_smac_test_out',
            delete_tmp_folder_after_terminate: bool = False
    ):
        self.pipeline = self.build_pipeline()

        self.pipeline_config = self.pipeline.get_default_pipeline_options()

        self.search_space = self.pipeline.get_hyperparameter_search_space()

        self._dataset_requirements: List[FitRequirement] = self.pipeline.get_dataset_requirements()
        self._stopwatch = StopWatch()
        self.task_type: Optional[str] = None
        self._automl: Optional[AutoMLSMBO] = None
        self.logging_config = logging_config
        self._temporary_directory = temporary_directory
        self._output_directory = output_directory
        self._delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self._backend = create(temporary_directory=self._temporary_directory,
                               output_directory=self._output_directory,
                               delete_output_folder_after_terminate=self._delete_tmp_folder_after_terminate)
        self._seed: Optional[int] = None
        self._metric: Optional[autoPyTorchMetric] = None
        self._logger: Optional[PicklableClientLogger] = None
        self._is_pipeline_fitted = False

    @abstractmethod
    def build_pipeline(self) -> BasePipeline:
        raise NotImplementedError

    def set_pipeline_config(
            self,
            **pipeline_config_kwargs: Any) -> None:
        """
        Check whether arguments are valid and then pipeline configuration.
        """
        unknown_keys = []
        for option, value in pipeline_config_kwargs.items():
            if option in self.pipeline_config.keys():
                pass
            else:
                unknown_keys.append(option)

        if len(unknown_keys) > 0:
            raise ValueError("Invalid configuration arguments given {},"
                             " expected arguments to be in {}".
                             format(unknown_keys, self.pipeline_config.keys()))

        self.pipeline_config.update(pipeline_config_kwargs)

    def get_pipeline_config(self) -> dict:
        """
        Returns the current pipeline configuration.
        """
        return self.pipeline_config

    def set_search_space(self, search_space: ConfigurationSpace) -> None:
        """
        Update the search space.
        """
        raise NotImplementedError

    def get_search_space(self) -> ConfigurationSpace:
        """
        Returns the current search space as ConfigurationSpace object.
        """
        return self.search_space

    def _get_logger(self, name):
        logger_name = 'AutoML:%s' % name

        # As AutoPyTorch works with distributed process,
        # we implement a logger server that can receive tcp
        # pickled messages. They are unpickled and processed locally
        # under the above logging configuration setting
        # We need to specify the logger_name so that received records
        # are treated under the logger_name ROOT logger setting
        context = multiprocessing.get_context('spawn')
        self.stop_logging_server = context.Event()
        port = context.Value('l')  # be safe by using a long
        port.value = -1

        self.logging_server = context.Process(
            target=start_log_server,
            kwargs=dict(
                host='localhost',
                logname=logger_name,
                event=self.stop_logging_server,
                port=port,
                filename='%s.log' % str(logger_name),
                logging_config=self.logging_config,
                output_dir=self._temporary_directory,
            ),
        )

        self.logging_server.start()

        while True:
            with port.get_lock():
                if port.value == -1:
                    time.sleep(0.01)
                else:
                    break

        self._logger_port = int(port.value)

        return get_named_client_logger(output_dir=self._temporary_directory,
                                       name=logger_name,
                                       port=port)

    def _clean_logger(self):
        if not hasattr(self, 'stop_logging_server') or self.stop_logging_server is None:
            return

        # Clean up the logger
        if self.logging_server.is_alive():
            self.stop_logging_server.set()

            # We try to join the process, after we sent
            # the terminate event. Then we try a join to
            # nicely join the event. In case something
            # bad happens with nicely trying to kill the
            # process, we execute a terminate to kill the
            # process.
            self.logging_server.join(timeout=5)
            self.logging_server.terminate()
            del self.stop_logging_server

    def _create_dask_client(self, n_jobs):
        self._is_dask_client_internally_created = True
        dask.config.set({'distributed.worker.daemon': False})
        self._dask_client = dask.distributed.Client(
            dask.distributed.LocalCluster(
                n_workers=n_jobs,
                processes=True,
                threads_per_worker=1,
                # We use the temporal directory to save the
                # dask workers, because deleting workers
                # more time than deleting backend directories
                # This prevent an error saying that the worker
                # file was deleted, so the client could not close
                # the worker properly
                local_directory=tempfile.gettempdir(),
                # Memory is handled by the pynisher, not by the dask worker/nanny
                memory_limit=0,
            ),
            # Heartbeat every 10s
            heartbeat_interval=10000,
        )

    def _close_dask_client(self):
        if (
                hasattr(self, '_is_dask_client_internally_created')
                and self._is_dask_client_internally_created
                and self._dask_client
        ):
            self._dask_client.shutdown()
            self._dask_client.close()
            del self._dask_client
            self._dask_client = None
            self._is_dask_client_internally_created = False
            del self._is_dask_client_internally_created

    def _load_models(self, resampling_strategy):
        self.ensemble_ = self._backend.load_ensemble(self._seed)

        # If no ensemble is loaded, try to get the best performing model
        if not self.ensemble_:
            self.ensemble_ = self._load_best_individual_model()

        if self.ensemble_:
            identifiers = self.ensemble_.get_selected_model_identifiers()
            self.models_ = self._backend.load_models_by_identifiers(identifiers)
            if isinstance(resampling_strategy, CrossValTypes):
                self.cv_models_ = self._backend.load_cv_models_by_identifiers(identifiers)
            else:
                self.cv_models_ = None
            # TODO: check for assertions in our case
            # if (
            #         len(self.models_) == 0 and
            #         self._resampling_strategy not in ['partial-cv', 'partial-cv-iterative-fit']
            # ):
            #     raise ValueError('No models fitted!')
            # if (
            #         self._resampling_strategy in ['cv', 'cv-iterative-fit']
            #         and len(self.cv_models_) == 0
            # ):
            #     raise ValueError('No models fitted!')

        elif self._disable_file_output or \
                (isinstance(self._disable_file_output, list) and
                 'pipeline' not in self._disable_file_output):
            model_names = self._backend.list_all_models(self._seed)

            if len(model_names) == 0:
                raise ValueError('No models fitted!')

            self.models_ = []

        else:
            self.models_ = []

    def _load_best_individual_model(self):
        """
        In case of failure during ensemble building,
        this method returns the single best model found
        by AutoML.
        This is a robust mechanism to be able to predict,
        even though no ensemble was found by ensemble builder.
        """

        # SingleBest contains the best model found by AutoML
        ensemble = SingleBest(
            metric=self._metric,
            seed=self._seed,
            run_history=self.run_history,
            backend=self._backend,
        )
        self._logger.warning(
            "No valid ensemble was created. Please check the log"
            "file for errors. Default to the best individual estimator:{}".format(
                ensemble.identifiers_
            )
        )
        return ensemble

    def _clean_search_run(self):
        if self._seed is not None:
            self._seed = None
        if self._metric is not None:
            self._metric = None
        if self.models_ is not None:
            self.models_ = None
        if self.cv_models_ is not None:
            self.cv_models_ = None
        if self.ensemble_ is not None:
            self.ensemble_ = None
        if self._logger is not None:
            self._logger = None
        if hasattr(self, 'run_history') and self.run_history is not None:
            del self.run_history
        if hasattr(self, 'trajectory') and self.trajectory is not None:
            del self.trajectory
        if hasattr(self, 'ensemble_performance_history') and self.ensemble_performance_history is not None:
            del self.ensemble_performance_history
        if hasattr(self, '_disable_file_output') and self._disable_file_output is not None:
            del self._disable_file_output

    @typing.no_type_check
    def search(
            self,
            dataset: BaseDataset,
            budget_type: Optional[str] = None,
            budget: Optional[float] = None,
            total_walltime_limit: int = 100,
            func_eval_time_limit: int = 60,
            memory_limit: Optional[int] = 3096,
            n_jobs: int = 2,
            seed: int = 1,
            include: Optional[Dict[str, Any]] = None,
            exclude: Optional[Dict[str, Any]] = None,
            disable_file_output: bool = False,
            smac_scenario_args: Optional[Dict[str, Any]] = None,
            get_smac_object_callback: Optional[Callable] = None,
            all_supported_metrics: bool = True,
            optimize_metric: Optional[str] = None,
            precision: int = 32,
            load_models: bool = True,
            ensemble_size: int = 1,
            ensemble_nbest: int = 1,
            max_models_on_disc: int = 1
    ):
        """
        Search for the best pipeline configuration for the given dataset
        using the optimizer.
        Args:
            dataset: (Dataset)
                The argument that will provide the dataset splits. It is
                a subclass of the  base dataset object which can
                generate the splits based on different restrictions.
            budget_type: (Optional[str])
                Type of budget to be used when fitting the pipeline.
                Either 'epochs' or 'runtime'. If not provided, uses
                the default in the pipeline config ('epochs')
            budget: (Optional[float])
                Budget to fit a single run of the pipeline. If not
                provided, uses the default in the pipeline config
        """
        self._clean_search_run()  # clean attributes created in previous search
        assert self.task_type == dataset.task_type, "Incompatible dataset entered for current task," \
                                                    "expected dataset to have task type :{} got " \
                                                    ":{}".format(self.task_type, dataset.task_type)
        # Initialise information needed for the experiment
        experiment_task_name = 'runSearch'
        self._stopwatch.start_task(experiment_task_name)
        self._seed = seed
        self._logger = self._get_logger(dataset.dataset_name)
        self._disable_file_output = disable_file_output
        # Save start time to backend
        self._backend.save_start_time(str(self._seed))

        self._backend.save_datamanager(dataset)

        dataset_properties = dataset.get_dataset_properties(self._dataset_requirements)
        self._metric = get_metrics(names=[optimize_metric], dataset_properties=dataset_properties)[0]

        budget_config = {}
        if budget_type is not None:
            assert budget is not None, "budget type was mentioned but not the budget to be used with it"
            budget_config['budget_type'] = budget_type
            budget_config[budget_type] = budget
        else:
            assert budget is None, "budget was mentioned but the budget type was not"

        self._create_dask_client(n_jobs=n_jobs)
        proc_ensemble = None
        if ensemble_size <= 0:
            self._logger.info("Not starting ensemble builder as ensemble size is 0")
        else:
            self._logger.info("Starting ensemble")
            ensemble_task_name = 'ensemble'
            self._stopwatch.start_task(ensemble_task_name)
            proc_ensemble = EnsembleBuilderManager(
                start_time=time.time(),
                time_left_for_ensembles=100,
                backend=copy.deepcopy(self._backend),
                dataset_name=dataset.dataset_name,
                output_type=STRING_TO_OUTPUT_TYPES[dataset.output_type],
                task_type=STRING_TO_TASK_TYPES[self.task_type],
                metrics=[self._metric],
                opt_metric=optimize_metric,
                ensemble_size=ensemble_size,
                ensemble_nbest=ensemble_nbest,
                max_models_on_disc=max_models_on_disc,
                seed=self._seed,
                max_iterations=1,
                read_at_most=np.inf,
                ensemble_memory_limit=memory_limit,
                random_state=self._seed,
                precision=precision,
            )
            self._stopwatch.stop_task(ensemble_task_name)

        # ==> Run SMAC
        smac_task_name = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(experiment_task_name)
        time_left_for_smac = max(0, total_walltime_limit - elapsed_time)

        self._logger.info("Starting SMAC with %5.2f sec time left" % time_left_for_smac)
        if time_left_for_smac <= 0:
            self._logger.warning(" Not starting SMAC becayse there is no time left")
        else:

            _proc_smac = AutoMLSMBO(
                config_space=self.get_search_space(),
                dataset_name=dataset.dataset_name,
                backend=self._backend,
                total_walltime_limit=total_walltime_limit,
                func_eval_time_limit=func_eval_time_limit,
                dask_client=self._dask_client,
                memory_limit=memory_limit,
                n_jobs=n_jobs,
                watcher=self._stopwatch,
                metric=self._metric,
                seed=self._seed,
                include=include,
                exclude=exclude,
                disable_file_output=disable_file_output,
                all_supported_metrics=all_supported_metrics,
                smac_scenario_args=smac_scenario_args,
                get_smac_object_callback=get_smac_object_callback,
                pipeline_config={**self.pipeline_config, **budget_config},
                ensemble_callback=proc_ensemble
            )
            try:
                self.run_history, self.trajectory, budget_type = \
                    _proc_smac.run_smbo()
                trajectory_filename = os.path.join(
                    self._backend.get_smac_output_directory_for_run(self._seed),
                    'trajectory.json')
                saveable_trajectory = \
                    [list(entry[:2]) + [entry[2].get_dictionary()] + list(entry[3:])
                     for entry in self.trajectory.trajectory]
                with open(trajectory_filename, 'w') as fh:
                    json.dump(saveable_trajectory, fh)
            except Exception as e:
                self._logger.exception(str(e))
                raise
        # Wait until the ensemble process is finished to avoid shutting down
        # while the ensemble builder tries to access the data
        self._logger.info("Starting Shutdown")

        if proc_ensemble is not None:
            self.ensemble_performance_history = list(proc_ensemble.history)

            # save the ensemble performance history file
            if len(self.ensemble_performance_history) > 0:
                pd.DataFrame(self.ensemble_performance_history).to_json(
                    os.path.join(self._backend.internals_directory, 'ensemble_history.json'))

            if len(proc_ensemble.futures) > 0:
                future = proc_ensemble.futures.pop()
                # Now we need to wait for the future to return as it cannot be cancelled while it
                # is running: https://stackoverflow.com/a/49203129
                self._logger.info("Ensemble script still running, waiting for it to finish.")
                future.result()
                self._logger.info("Ensemble script finished, continue shutdown.")

        if load_models:
            self._logger.info("Loading models...")
            self._load_models(dataset.resampling_strategy)
            self._logger.info("Finished loading models...")

        self._logger.info("Closing the dask infrastructure")
        self._close_dask_client()
        self._logger.info("Finished closing the dask infrastructure")

        # Clean up the logger
        self._logger.info("Starting to clean up the logger")
        self._clean_logger()

        return self

    @typing.no_type_check
    def fit(
            self,
            dataset: BaseDataset,
            budget_type: str,
            budget: int,
    ):
        """Refit a model configuration and calculate the model performance.
        Given a model configuration, the model is trained on the joint train
        and validation sets of the dataset. This corresponds to the refit
        phase after finding a best hyperparameter configuration after the hpo
        phase.
        Args:
            dataset: (Dataset)
                The argument that will provide the dataset splits. It can either
                be a dictionary with the splits, or the dataset object which can
                generate the splits based on different restrictions.
            model_config: (Configuration)
                The configuration of the model.
        Returns:
            Value of the evaluation metric calculated on the test set.
        """
        budget_config = {}
        if budget_type is not None:
            assert budget is not None, "budget type was mentioned but not the budget to be used with it"
            budget_config['budget_type'] = budget_type
            budget_config[budget_type] = budget
        else:
            assert budget is None, "budget was mentioned but the budget type was not"

        dataset_properties = dataset.get_dataset_properties(self._dataset_requirements)

        X: Dict[str, Any] = dict({'dataset_properties': dataset_properties})
        X.update({**self.pipeline_config, **budget_config})
        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models(dataset.resampling_strategy)

        # Refit is not applicable when ensemble_size is set to zero.
        if self.ensemble_ is None:
            raise ValueError("Refit can only be called if 'ensemble_size != 0'")

        for identifier in self.models_:
            model = self.models_[identifier]
            # this updates the model inplace, it can then later be used in
            # predict method

            # try to fit the model. If it fails, shuffle the data. This
            # could alleviate the problem in algorithms that depend on
            # the ordering of the data.
            fit_and_suppress_warnings(self._logger, model, X, y=None)

        self._is_pipeline_fitted = True

        return self

    def predict(
            self,
            X_test: np.ndarray,
            batch_size: Optional[int] = None,
            n_jobs: int = 1
    ) -> np.ndarray:
        """Generate the estimator predictions.
        Generate the predictions based on the given examples from the test set.
        Args:
        X_test: (np.ndarray)
            The test set examples.
        Returns:
            Array with estimator predictions.
        """
        assert self._is_pipeline_fitted, "Can't predict if the pipeline has not been fitted yet"
        # Parallelize predictions across models with n_jobs processes.
        # Each process computes predictions in chunks of batch_size rows.
        try:
            for i, tmp_model in enumerate(self.models_.values()):
                if isinstance(tmp_model, (DummyRegressor, DummyClassifier)):
                    check_is_fitted(tmp_model)
                else:
                    check_is_fitted(tmp_model.steps[-1][-1])
            models = self.models_
        except sklearn.exceptions.NotFittedError:
            # When training a cross validation model, self.cv_models_
            # will contain the Voting classifier/regressor product of cv
            # self.models_ in the case of cv, contains unfitted models
            # Raising above exception is a mechanism to detect which
            # attribute contains the relevant models for prediction
            try:
                check_is_fitted(list(self.cv_models_.values())[0])
                models = self.cv_models_
            except sklearn.exceptions.NotFittedError:
                raise ValueError('Found no fitted models!')

        all_predictions = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_model_predict)(
                models[identifier], X_test, batch_size, self._logger, self.task_type
            )
            for identifier in self.ensemble_.get_selected_model_identifiers()
        )

        if len(all_predictions) == 0:
            raise ValueError('Something went wrong generating the predictions. '
                             'The ensemble should consist of the following '
                             'models: %s, the following models were loaded: '
                             '%s' % (str(list(self.ensemble_.indices_.keys())),
                                     str(list(self.models_.keys()))))

        predictions = self.ensemble_.predict(all_predictions)

        if self.task_type not in REGRESSION_TASKS:
            # Make sure average prediction probabilities
            # are within a valid range
            # Individual models are checked in _model_predict
            predictions = np.clip(predictions, 0.0, 1.0)

        return predictions

    # def score(
    #         self,
    #         X_test: np.ndarray,
    #         y_test: np.ndarray,
    #         sample_weights: Optional[np.ndarray] = None,
    # ) -> float:
    #     """Calculate the score on the test set.
    #     Calculate the evaluation measure on the test set.
    #     Args:
    #     X_test: (np.ndarray)
    #         The test examples of the dataset.
    #     y_test: (np.ndarray)
    #         The test ground truth labels.
    #     sample_weights: (np.ndarray|None)
    #         The weights for each sample.
    #     Returns:
    #         Value of the evaluation metric calculated on the test set.
    #     """
    #     assert self._is_pipeline_fitted, "Can't calculate score if the pipeline has not been fitted yet"
    #     return self.pipeline.score(X_test, y_test, sample_weights)

    @typing.no_type_check
    def get_incumbent_results(
            self
    ):
        pass

    @typing.no_type_check
    def get_incumbent_config(
            self
    ):
        pass
