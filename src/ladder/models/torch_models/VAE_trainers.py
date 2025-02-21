"""The VAE trainers module houses trainer classes used with VAE variants."""

import copy
import warnings
from typing import Literal

import numpy as np
import pyro
import pyro.optim as opt
import torch
import torch.utils.data as utils
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.poutine import uncondition
from torch import nn


class _BasePyroTrainerMixin:
    """Mixin for basic Pyro models. All losses should be available through a pass of model & guide. Do NOT instantiate.

    Parameters
    ----------
    model : :class:`~torch.nn.Module`
        The model to train.

    train_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the training set.

    test_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the test set.

    optim : :class:`~pyro.optim.PyroOptim`, default: None
        Optimizer to be used for training. Defaults to :class:`~pyro.optim.Adam` with learning rate 1e-3 and default parameters.

    verbose : `bool`, default: False
        Determine whether to print losses after every epoch

    """

    @staticmethod
    def _send_args_to_device(args, device):
        return tuple(
            [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        )

    def __init__(
        self,
        model: nn.Module,
        train_loader: utils.DataLoader,
        test_loader: utils.DataLoader,
        optim=opt.Adam({"lr": 1e-3}),
        verbose: bool = True,
    ):
        self.reset()
        self.train_losses, self.test_losses, self.epochs, self.predictive = (
            [],
            [],
            0,
            None,
        )
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader, self.test_loader = train_loader, test_loader
        self.model, self.elbo, self.optim = model.to(self.device), Trace_ELBO(), optim
        self.svi = SVI(self.model.model, self.model.guide, self.optim, self.elbo)

    def get_variables(self, which: Literal["train", "test"] = "train"):
        try:
            assert self.predictive is not None
        except AssertionError:
            warnings.warn(
                "Predictive missing, please run training before calling the trainer for predictions, returning None.",
                stacklevel=2,
            )
            return None

        match which:
            case "train":
                return self.predictive(
                    *self._send_args_to_device(
                        self.train_loader.dataset[:], self.device
                    )
                )

            case "test":
                return self.predictive(
                    *self._send_args_to_device(self.test_loader.dataset[:], self.device)
                )

            case _:
                warnings.warn(
                    'Invalid args for "which". Please enter either "train" or "test", returning None.',
                    stacklevel=2,
                )
                return None

    def _predictive_setup(self):
        try:
            self.predictive = Predictive(
                uncondition(self.best_model.model),
                guide=self.best_model.guide,
                num_samples=1,
            )

        except AttributeError:
            if self.verbose:
                print("Best model not tracked, defaulting to model from last epoch.")

            self.predictive = Predictive(
                uncondition(self.model.model), guide=self.model.guide, num_samples=1
            )

    def train_single_epoch(self):
        """Trains the attached model for a single pass through the dataset."""
        self.model.train()

        train_losses, test_losses = [], []

        for args in self.train_loader:
            args = self._send_args_to_device(args, self.device)
            train_losses.append(self.svi.step(*args))

        self.model.eval()
        with torch.no_grad():
            for args in self.test_loader:
                args = self._send_args_to_device(args, self.device)
                test_losses.append(
                    self.elbo.loss(self.model.model, self.model.guide, *args)
                )

        if self.verbose:
            print(
                f"Epoch : {self.epochs + 1} || Train Loss: {np.mean(train_losses).round(3)} || Test Loss: {np.mean(test_losses).round(3)}"
            )

        self.train_losses.append(np.mean(train_losses))
        self.test_losses.append(np.mean(test_losses))
        self.epochs += 1

    def reset(self):
        """Resets Pyro parameter storage for continued training."""
        pyro.clear_param_store()
        self.train_losses, self.test_losses, self.epochs = [], [], 0

    def train(self):
        """Trains the attached model until the designated stop condition is reached."""
        while not self.is_stop_condition():
            self.train_single_epoch()

        self._predictive_setup()

    def is_stop_condition(self):
        """Defines the stop condition for the model."""
        pass


class _AdversarialPyroTrainerMixin(_BasePyroTrainerMixin):
    """Mixin for Pyro models with adversarial loss (CSVAE). Do NOT instantiate."""

    def __init__(
        self,
        *args,
    ):
        _BasePyroTrainerMixin.__init__(self, *args)
        params = dict(self.model.named_parameters())

        classifier_params, model_params = [], []

        for k, v in params.items():
            if "classifiers" in k:
                classifier_params.append(v)

            else:
                model_params.append(v)

        self.model_opt = self.optim.pt_optim_constructor(
            model_params, **self.optim.pt_optim_args
        )
        self.classifier_opt = self.optim.pt_optim_constructor(
            classifier_params, **self.optim.pt_optim_args
        )
        self.svi = None

    def _train_step_classifier(self, *args):
        self.classifier_opt.zero_grad()
        self.model.classification(*args).mean().backward()
        self.classifier_opt.step()

    def _train_step_model(self, *args):
        loss = self.elbo.differentiable_loss(self.model.model, self.model.guide, *args)
        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

        return loss.detach().cpu()

    def train_single_epoch(self):
        """Trains the attached model with adversarial loss for a single pass through the dataset."""
        self.model.train()

        train_losses, test_losses = [], []

        for args in self.train_loader:
            args = self._send_args_to_device(args, self.device)

            self._train_step_classifier(*args)
            train_losses.append(self._train_step_model(*args))

        self.model.eval()
        with torch.no_grad():
            for args in self.test_loader:
                args = self._send_args_to_device(args, self.device)
                test_losses.append(
                    self.elbo.differentiable_loss(
                        self.model.model, self.model.guide, *args
                    )
                    .detach()
                    .cpu()
                )

        if self.verbose:
            print(
                "Epoch :",
                self.epochs + 1,
                "|| Train Loss:",
                np.mean(train_losses).round(3),
                "|| Test Loss:",
                np.mean(test_losses).round(3),
            )

        self.train_losses.append(np.mean(train_losses))
        self.test_losses.append(np.mean(test_losses))
        self.epochs += 1


class _EpochMixin:
    """Adds epoch based stopping capability."""

    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def is_stop_condition(self):
        """Stop upon reaching the designated number of epochs."""
        return self.epochs >= self.max_epochs


class _ThresholdMixin:
    """Adds threshold based stopping capability"""

    def __init__(self, convergence_threshold, patience):
        self.convergence_threshold = convergence_threshold
        self.patience, self.max_patience = 0, patience
        self.best_model = copy.deepcopy(self.model)

    def is_stop_condition(self):
        """Stop when patience runs out without improvement."""
        if not self.patience < self.max_patience:
            return True

        try:

            if (
                min(self.test_losses[:-1]) - self.test_losses[-1]
                > self.convergence_threshold
            ):
                self.patience = 0
                del self.best_model
                self.best_model = copy.deepcopy(self.model)

            else:
                self.patience += 1

        except ValueError:
            pass

        return False


class EpochPyroTrainer(_EpochMixin, _BasePyroTrainerMixin):
    """Trainer class that stops upon reaching the designated number of epochs.

    Parameters
    ----------
    max_epochs : int
        Number of epochs to run the model for.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, max_epochs: int, *args):
        _BasePyroTrainerMixin.__init__(self, *args)
        _EpochMixin.__init__(self, max_epochs)


class ThresholdPyroTrainer(_ThresholdMixin, _BasePyroTrainerMixin):
    """Trainer class that stops upon reaching the designated number of epochs.

    Parameters
    ----------
    convergence_threshold : float, default: 1e-3
        Minimum improvement required to keep the model running.

    patience : int, default: 15
        Number of epochs allowed for minimum improvement to be observed.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, convergence_threshold: float = 1e-3, patience: int = 15, *args):
        _BasePyroTrainerMixin.__init__(self, *args)
        _ThresholdMixin.__init__(self, convergence_threshold, patience)

    def reset(self):
        """Resets Pyro parameter storage for continued training."""
        _BasePyroTrainerMixin.reset(self)
        self.best_model = None
        self.patience = 0


class AdversarialEpochPyroTrainer(_EpochMixin, _AdversarialPyroTrainerMixin):
    """Adversarial trainer class that stops upon reaching the designated number of epochs.

    Parameters
    ----------
    max_epochs : int
        Number of epochs to run the model for.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, max_epochs: int, *args):
        _AdversarialPyroTrainerMixin.__init__(self, *args)
        _EpochMixin.__init__(self, max_epochs)


class AdversarialThresholdPyroTrainer(_ThresholdMixin, _AdversarialPyroTrainerMixin):
    """Adversarial trainer class that stops upon reaching the designated number of epochs.

    Parameters
    ----------
    convergence_threshold : float, default: 1e-3
        Minimum improvement required to keep the model running.

    patience : int, default: 15
        Number of epochs allowed for minimum improvement to be observed.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, convergence_threshold: float = 1e-3, patience: int = 15, *args):
        _AdversarialPyroTrainerMixin.__init__(self, *args)
        _ThresholdMixin.__init__(self, convergence_threshold, patience)

    def reset(self):
        """Resets Pyro parameter storage for continued training."""
        _AdversarialPyroTrainerMixin.reset(self)
        self.patience = 0
