"""The VAE trainers module houses trainer classes used with VAE variants."""

import numpy as np
import pyro
import pyro.optim as opt
import torch
import torch.utils.data as utils
from pyro.infer import SVI, Trace_ELBO
from torch import nn


class BasePyroTrainerMixin:
    """Mixin for basic Pyro models. All losses should be available through a pass of model & guide. Do NOT instantiate.

    Parameters
    ----------
    model : :class:`~torch.nn.Module`
        The model to train.

    train_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the training set.

    test_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the test set.

    opt : :class:`~pyro.optim.PyroOptim`, default: None
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
        opt=opt.Adam({"lr": 1e-3}),
        verbose: bool = True,
    ):
        self.reset()
        self.train_losses, self.test_losses, self.epochs = [], [], 0
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader, self.test_loader = train_loader, test_loader
        self.model, self.elbo, self.opt = model.to(self.device), Trace_ELBO(), opt
        self.svi = SVI(self.model.model, self.model.guide, self.opt, self.elbo)

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
                f"Epoch : {self.epochs + 1} || Train Loss: {np.mean(train_losses).round(5)} || Test Loss: {np.mean(test_losses).round(5)}"
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

    def is_stop_condition(self):
        """Defines the stop condition for the model."""
        pass


class EpochPyroTrainer(BasePyroTrainerMixin):
    """Trainer class that stops upon reaching the designated number of epochs.

    Parameters
    ----------
    max_epochs : int
        Number of epochs to run the model for.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, max_epochs: int, *args):
        BasePyroTrainerMixin.__init__(self, *args)
        self.max_epochs = max_epochs

    def is_stop_condition(self):
        """Stop upon reaching the designated number of epochs."""
        return self.epochs >= self.max_epochs


class ThresholdPyroTrainer(BasePyroTrainerMixin):
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

        BasePyroTrainerMixin.__init__(self, *args)
        self.convergence_threshold = convergence_threshold
        self.patience, self.max_patience = 0, patience

    def reset(self):
        """Resets Pyro parameter storage for continued training."""
        BasePyroTrainerMixin.reset(self)
        self.patience = 0

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

            else:
                self.patience += 1

        except ValueError:
            pass

        return False
