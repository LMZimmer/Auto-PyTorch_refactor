import collections
import logging.handlers
import os
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import numpy as np

import pynisher

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.training.losses import get_loss_instance
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent,
    BudgetTracker,
    RunSummary,
)
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.logging_ import get_named_client_logger

trainer_directory = os.path.split(__file__)[0]
_trainers = find_components(__package__,
                            trainer_directory,
                            BaseTrainerComponent)
_addons = ThirdPartyComponents(BaseTrainerComponent)


def add_trainer(trainer: BaseTrainerComponent) -> None:
    _addons.add_component(trainer)


class TrainerChoice(autoPyTorchChoice):
    """This class is an interface to the PyTorch trainer.


    To map to pipeline terminology, a choice component will implement the epoch
    loop through fit, whereas the component who is chosen will dictate how a single
    epoch happens, that is, how batches of data are fed and used to train the network.

    """
    def __init__(self,
                 dataset_properties: Dict[str, Any],
                 random_state: Optional[np.random.RandomState] = None
                 ):

        super().__init__(dataset_properties=dataset_properties,
                         random_state=random_state)
        self.run_summary = None  # type: Optional[RunSummary]
        self.writer = None  # type: Optional[SummaryWriter]
        self._fit_requirements: Optional[List[FitRequirement]] = [
            FitRequirement("lr_scheduler", (_LRScheduler,), user_defined=False, dataset_property=False),
            FitRequirement("network", (torch.nn.Sequential,), user_defined=False, dataset_property=False),
            FitRequirement(
                "optimizer", (Optimizer,), user_defined=False, dataset_property=False),
            FitRequirement("train_data_loader",
                           (torch.utils.data.DataLoader,),
                           user_defined=False, dataset_property=False),
            FitRequirement("val_data_loader",
                           (torch.utils.data.DataLoader,),
                           user_defined=False, dataset_property=False)]

    def get_fit_requirements(self) -> Optional[List[FitRequirement]]:
        return self._fit_requirements

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available trainer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all components available
                as choices for learning rate scheduling
        """
        components = collections.OrderedDict()  # type: Dict[str, autoPyTorchComponent]
        components.update(_trainers)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(
        self,
        dataset_properties: Optional[Dict[str, str]] = None,
        default: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            default (Optional[str]): Default scheduler to use
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal trainers for this problem
        available_trainers = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_trainers) == 0:
            raise ValueError("No trainer found")

        if default is None:
            defaults = ['StandartTrainer',
                        ]
            for default_ in defaults:
                if default_ in available_trainers:
                    default = default_
                    break

        trainer = CategoricalHyperparameter(
            '__choice__',
            list(available_trainers.keys()),
            default_value=default
        )
        cs.add_hyperparameter(trainer)
        for name in available_trainers:
            trainer_configuration_space = available_trainers[name]. \
                get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': trainer, 'value': name}
            cs.add_configuration_space(
                name,
                trainer_configuration_space,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'run_summary': self.run_summary})
        return X

    def fit(self, X: Dict[str, Any], y: Any = None, **kwargs: Any) -> autoPyTorchComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        # Make sure that the prerequisites are there
        self.check_requirements(X, y)

        # Setup the logger
        self.logger = get_named_client_logger(
            output_dir=X['backend'].temporary_directory,
            name=X['job_id'],
            # Log to a user provided port else to the default logging port
            port=X['logger_port'
                   ] if 'logger_port' in X else logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        )

        fit_function = self._fit
        if X['use_pynisher']:
            wall_time_in_s = X['runtime'] if 'runtime' in X else None
            memory_limit = X['cpu_memory_limit'] if 'cpu_memory_limit' in X else None
            fit_function = pynisher.enforce_limits(
                wall_time_in_s=wall_time_in_s,
                mem_in_mb=memory_limit,
                logger=self.logger
            )(self._fit)

        # Call the actual fit function.
        state_dict = fit_function(
            X=X,
            y=y,
            **kwargs
        )

        if X['use_pynisher']:
            # Normally the X[network] is a pointer to the object, so at the
            # end, when we train using X, the pipeline network is updated for free
            # If we do multiprocessing (because of pynisher) we have to update
            # X[network] manually. we do so in a way that every pipeline component
            # can see this new network -- via an update, not overwrite of the pointer
            state_dict = state_dict.result
            X['network'].load_state_dict(state_dict)

        # TODO: when have the optimizer code, the pynisher object might have failed
        # We should process this function as Failure if so trough fit_function.exit_status
        return cast(autoPyTorchComponent, self.choice)

    def _fit(self, X: Dict[str, Any], y: Any = None, **kwargs: Any) -> torch.nn.Module:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Comply with mypy
        # Notice that choice here stands for the component choice framework,
        # where we dynamically build the configuration space by selecting the available
        # component choices. In this case, is what trainer choices are available
        assert self.choice is not None

        # Setup a Logger and other logging support
        # Writer is not pickable -- make sure it is not saved in self
        writer = None
        if 'use_tensorboard_logger' in X and X['use_tensorboard_logger']:
            writer = SummaryWriter(log_dir=X['backend'].temporary_directory)

        if X["torch_num_threads"] > 0:
            torch.set_num_threads(X["torch_num_threads"])

        budget_tracker = BudgetTracker(
            budget_type=X['budget_type'],
            max_runtime=X['runtime'] if 'runtime' in X else None,
            max_epochs=X['epochs'] if 'epochs' in X else None,
        )

        # Support additional user metrics
        additional_metrics = X['additional_metrics'] if 'additional_metrics' in X else None
        additional_losses = X['additional_losses'] if 'additional_losses' in X else None
        self.choice.prepare(
            model=X['network'],
            metrics=get_metrics(dataset_properties=X['dataset_properties'],
                                names=additional_metrics),
            criterion=get_loss_instance(X['dataset_properties'],
                                        name=additional_losses),
            budget_tracker=budget_tracker,
            optimizer=X['optimizer'],
            device=self.get_device(X),
            metrics_during_training=X['metrics_during_training'],
            scheduler=X['lr_scheduler'],
            task_type=STRING_TO_TASK_TYPES[X['dataset_properties']['task_type']]
        )
        total_parameter_count, trainable_parameter_count = self.count_parameters(X['network'])
        self.run_summary = RunSummary(
            total_parameter_count,
            trainable_parameter_count,
        )

        epoch = 1

        while True:

            # prepare epoch
            start_time = time.time()

            self.choice.on_epoch_start(X=X, epoch=epoch)

            # training
            train_loss, train_metrics = self.choice.train_epoch(
                train_loader=X['train_data_loader'],
                epoch=epoch,
                logger=self.logger,
                writer=writer,
            )

            val_loss, val_metrics, test_loss, test_metrics = None, {}, None, {}
            if self.eval_valid_each_epoch(X):
                val_loss, val_metrics = self.choice.evaluate(X['val_data_loader'], epoch, writer)
                if 'test_data_loader' in X and X['test_data_loader']:
                    test_loss, test_metrics = self.choice.evaluate(X['test_data_loader'], epoch, writer)

            # Save training information
            self.run_summary.add_performance(
                epoch=epoch,
                start_time=start_time,
                end_time=time.time(),
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )

            # Save the weights of the best model and, if patience
            # exhausted break training
            if self.early_stop_handler(X):
                break

            if self.choice.on_epoch_end(X=X, epoch=epoch):
                break

            self.logger.debug(self.run_summary.repr_last_epoch())

            # Reached max epoch on next iter, don't even go there
            if budget_tracker.is_max_epoch_reached(epoch + 1):
                break

            epoch += 1

            torch.cuda.empty_cache()

        # wrap up -- add score if not evaluating every epoch
        if not self.eval_valid_each_epoch(X):
            val_loss, val_metrics = self.choice.evaluate(X['val_data_loader'])
            if 'test_data_loader' in X and X['val_data_loader']:
                test_loss, test_metrics = self.choice.evaluate(X['test_data_loader'])
            self.run_summary.add_performance(
                epoch=epoch,
                start_time=start_time,
                end_time=time.time(),
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
            self.logger.debug(self.run_summary.repr_last_epoch())
            self.save_model_for_ensemble()

        self.logger.info(f"Finished training with {self.run_summary.repr_last_epoch()}")

        # Tag as fitted
        self.fitted_ = True

        return X['network'].state_dict()

    def early_stop_handler(self, X: Dict[str, Any]) -> bool:
        """
        If early stopping is enabled, this procedure stops the training after a
        given patience
        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted

        Returns:
            bool: If true, training should be stopped
        """
        assert self.run_summary is not None
        epochs_since_best = self.run_summary.get_best_epoch() - self.run_summary.get_last_epoch()
        if epochs_since_best > X['early_stopping']:
            return True

        return False

    def eval_valid_each_epoch(self, X: Dict[str, Any]) -> bool:
        """
        Returns true if we are supposed to evaluate the model on every epoch,
        on the validation data. Usually, we only validate the data at the end,
        but in the case of early stopping, is appealing to evaluate each epoch.
        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted

        Returns:
            bool: if True, the model is evaluated in every epoch

        """
        if 'early_stopping' in X and X['early_stopping']:
            return True

        # We need to know if we should reduce the rate based on val loss
        if 'ReduceLROnPlateau' in X['lr_scheduler'].__class__.__name__:
            return True

        return False

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """

        # make sure the parent requirements are honored
        super().check_requirements(X, y)

        # We need a working dir in where to put our data
        if 'backend' not in X:
            raise ValueError('Need a backend to provide the working directory, '
                             "yet 'backend' was not found in the fit dictionary")

        # For resource allocation, we need to know if pynisher is enabled
        if 'use_pynisher' not in X:
            raise ValueError('Missing use_pynisher in the fit dictionary')

        # Whether we should evaluate metrics during training or no
        if 'metrics_during_training' not in X:
            raise ValueError('Missing metrics_during_training in the fit dictionary')

        # Setup Components
        if 'lr_scheduler' not in X:
            raise ValueError("Learning rate scheduler not found in the fit dictionary!")

        if 'network' not in X:
            raise ValueError("Network not found in the fit dictionary!")

        if 'optimizer' not in X:
            raise ValueError("Optimizer not found in the fit dictionary!")

        # Training Components
        if 'train_data_loader' not in X:
            raise ValueError("train_data_loader not found in the fit dictionary!")

        if 'val_data_loader' not in X:
            raise ValueError("val_data_loader not found in the fit dictionary!")

        if 'budget_type' not in X:
            raise ValueError("Budget type not found in the fit dictionary!")
        else:
            if 'epochs' not in X or 'runtime' not in X or 'epoch_or_time' not in X:
                if X['budget_type'] in ['epochs', 'epoch_or_time'] and 'epochs' not in X:
                    raise ValueError("Budget type is epochs but "
                                     "no epochs was not found in the fit dictionary!")
                elif X['budget_type'] in ['runtime', 'epoch_or_time'] and 'runtime' not in X:
                    raise ValueError("Budget type is runtime but "
                                     "no maximum number of seconds was provided!")
            else:
                raise ValueError("Unsupported budget type provided: {}".format(
                    X['budget_type']
                ))

        if 'job_id' not in X:
            raise ValueError('Need a job identifier to be able to isolate jobs')

        for config_option in ["torch_num_threads", 'device']:
            if config_option not in X:
                raise ValueError("Missing config option {} in config".format(
                    config_option
                ))

    def get_device(self, X: Dict[str, Any]) -> torch.device:
        """
        Returns the device to do torch operations

        Args:
            X (Dict[str, Any]): A fit dictionary to control how the pipeline
                is fitted

        Returns:
            torch.device: the device in which to compute operations. Cuda/cpu
        """
        if not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(X['device'])

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        """
        A method to get the total/trainable parameter count from the model

        Args:
            model (torch.nn.Module): the module from which to count parameters

        Returns:
            total_parameter_count: the total number of parameters of the model
            trainable_parameter_count: only the parameters being optimized
        """
        total_parameter_count = sum(
            p.numel() for p in model.parameters())
        trainable_parameter_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        return total_parameter_count, trainable_parameter_count

    def save_model_for_ensemble(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = str(self.run_summary)
        return string
