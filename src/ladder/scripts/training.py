"""The training module includes the functions used to train the models.

The functions defined here can be run independently for low-level
specific applications, or through the workflows API for high-level
standard applications.
"""

import numpy as np
import pyro
import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as utils
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import MultiStepLR
from tqdm import tqdm


# Helper to get device
def get_device():
    """Prints currently used device.

    Returns
    -------
    torch.device
        Device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to train Pyro models
def train_pyro(
    model: nn.Module,
    train_loader: utils.DataLoader,
    test_loader: utils.DataLoader,
    num_epochs: int = 500,
    convergence_threshold: float = 1e-3,
    convergence_window: int = 15,
    verbose: bool = True,
    device: torch.device = get_device(),
    optim_args: dict = None,
):
    """Runner for basic Pyro models.

    Trains up to `num_epochs` or until a new minimum is not attained
    that is lower than the older minimum by `convergence_threshold` for
    `convergence_window` epochs.

    Parameters
    ----------
    model : nn.module
        The model to train.

    train_loader : utils.DataLoader
        Data loader for the training set.

    test_loader : utils.DataLoader
        Data loader for the test set.

    num_epochs : int, default: 500
        Maximum number of epochs to run.

    convergence_threshold : float, default: 1e-3
        Minimum improvement to decide on convergence.

    convergence_window : int, default: 15
        Patience window for deciding on convergence.

    verbose : bool, default: True
        If `True`, prints out the loss at every epoch.

    device : torch.device
        Device object to run models on.

    optim_args : dict, default: {"optimizer": opt.Adam,"optim_args": {"lr": 1e-3, "eps": 1e-2},"gamma": 1,"milestones": [1e10]}
        Arguments to be passed to `MultiStepLR` for fine tuning if needed.

    Returns
    -------
    model : nn.Module
        The model object post-training.

    loss_track_train : np.array
        `float` array containing the training loss per epoch.

    loss_track_test : np.array
        `float` array containing the test loss per epoch.
    """
    if optim_args is None:
        optim_args = {
            "optimizer": opt.Adam,
            "optim_args": {"lr": 1e-2, "eps": 1e-2},
            "gamma": 1,
            "milestones": [1e10],
        }
    print(f"Using device: {device}\n")

    model = model.double().to(device)
    scheduler = MultiStepLR(optim_args.copy())
    elbo = Trace_ELBO()
    svi = SVI(model.model, model.guide, scheduler, elbo)

    loss_track_test, loss_track_train, losses_min = [], [], [np.inf]
    min_count = 0

    if verbose:
        num_epochs = range(num_epochs)
    else:
        num_epochs = tqdm(range(num_epochs))

    for epoch in num_epochs:
        losses = []
        losses_test = []

        model.train()

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            loss = svi.step(x, y)
            losses.append(loss)

        model.eval()
        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss = elbo.loss(model.model, model.guide, x, y)
                losses_test.append(test_loss)

        scheduler.step()

        if verbose:
            print(
                f"Epoch : {epoch} || Train Loss: {np.mean(losses).round(5)} || Test Loss: {np.mean(losses_test).round(5)}"
            )

        loss_track_train.append(np.mean(losses))
        loss_track_test.append(np.mean(losses_test))
        min_count += 1

        if (np.min(losses_min) - np.mean(losses_test)) > convergence_threshold:
            losses_min.append(np.mean(losses_test))
            min_count = 0

        if min_count == convergence_window:
            print(
                f"Convergence detected with last {convergence_window} epochs improvement {losses_min[-1] - np.min(loss_track_test[-convergence_window:])}, ending training..."
            )
            break

    return model, loss_track_train, loss_track_test


# Helper to train models that involve disjoint parameters during training
def train_pyro_disjoint_param(
    model: nn.Module,
    train_loader: utils.DataLoader,
    test_loader: utils.DataLoader,
    num_epochs: int = 500,
    convergence_threshold: float = 1e-3,
    convergence_window: int = 15,
    verbose: bool = True,
    device: torch.device = get_device(),
    lr: float = 1e-2,
    eps: float = 1e-2,
    betas: tuple = (0.90, 0.999),
    warmup: int = 0,
):
    """Runner for Patches, but can be used for other adversarial models.

    Trains up to `num_epochs` or until a new minimum is not attained
    that is lower than the older minimum by `convergence_threshold` for
    `convergence_window` epochs. Allows for setting different training routines
    for the adversarial loss.

    Parameters
    ----------
    model : nn.module
        The model to train.

    train_loader : utils.DataLoader
        Data loader for the training set.

    test_loader : utils.DataLoader
        Data loader for the test set.

    num_epochs : int, default: 500
        Maximum number of epochs to run.

    convergence_threshold : float, default: 1e-3
        Minimum improvement to decide on convergence.

    convergence_window : int, default: 15
        Patience window for deciding on convergence.

    verbose : bool, default: True
        If `True`, prints out the loss at every epoch.

    device : torch.device
        Device object to run models on.

    lr : float, default: 1e-2
        Learning rate for the model.

    eps : float, default: 1e-2
        Eps to be passed to the Adam optimizer.

    betas : tuple, default: (0.90, 0.999)
        Betas to be passed to the Adam optimizer.

    warmup : int, default: 0
        Number of epochs to run the classifier before running the entire model.

    Returns
    -------
    model : nn.Module
        The model object post-training.

    loss_track_train : np.array
        `float` array containing the training loss per epoch.

    loss_track_test : np.array
        `float` array containing the test loss per epoch.

    params_nonc_names : set(str)
        `str` set containing model parameter names except the classifier.

    params_c_names : set(str)
        `str` set containing model parameter names for the classifier.
    """
    print(f"Using device: {device}\n")

    model = model.double().to(device)
    loss_track_test, loss_track_train, losses_min = [], [], [np.inf]
    min_count = 0

    # Defining losses
    loss_fn = lambda model, guide, x, y: pyro.infer.Trace_ELBO().differentiable_loss(
        model, guide, x, y
    )

    # Params & optims
    x, y, _ = next(iter(train_loader))
    with pyro.poutine.trace(param_only=True) as param_capture:
        loss = loss_fn(model.model, model.guide, x.to(device), y.to(device))

    params_nonc = {
        site["value"].unconstrained()
        for site in param_capture.trace.nodes.values()
        if "classifier_z" not in site["name"]
    }
    params_nonc_names = {
        site["name"]
        for site in param_capture.trace.nodes.values()
        if "classifier_z" not in site["name"]
    }

    params_c = {
        site["value"].unconstrained()
        for site in param_capture.trace.nodes.values()
        if "classifier_z" in site["name"]
    }
    params_c_names = {
        site["name"]
        for site in param_capture.trace.nodes.values()
        if "classifier_z" in site["name"]
    }

    optimizer_nonc = torch.optim.Adam(params_nonc, lr=lr, eps=eps, betas=betas)
    optimizer_c = torch.optim.Adam(params_c, lr=lr, eps=eps, betas=betas)

    # Train loop

    if verbose:
        num_epochs = range(num_epochs)
    else:
        num_epochs = tqdm(range(num_epochs))

    for epoch in num_epochs:
        losses, prob_losses = [], []
        losses_test, prob_losses_test = [], []

        model.train()

        # Classifier trains over dataset
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            log_prob_loss = model.adversarial(x, y).mean()
            prob_losses.append(log_prob_loss.detach().cpu())

            optimizer_c.zero_grad()
            log_prob_loss.backward()
            optimizer_c.step()

        # Other parameters also train
        if epoch + 1 > warmup:
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model.model, model.guide, x, y)
                losses.append(loss.detach().cpu())

                optimizer_nonc.zero_grad()
                loss.backward()
                optimizer_nonc.step()

        # Testing
        model.eval()

        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss = loss_fn(model.model, model.guide, x, y)
                losses_test.append(test_loss.detach().cpu())

                log_prob_loss = model.adversarial(x, y).mean()
                prob_losses_test.append(log_prob_loss.detach().cpu())

        if verbose:
            print(
                f"Epoch : {epoch} || Train Loss: {np.mean(losses).round(5)} // {np.mean(prob_losses).round(5)} || Test Loss: {np.mean(losses_test).round(5)} // {np.mean(prob_losses_test).round(5)} || Warmup : {bool(epoch+1 <= warmup)}"
            )

        loss_track_train.append(np.mean(losses))
        loss_track_test.append(np.mean(losses_test))

        min_count += 1

        if (np.min(losses_min) - np.mean(losses_test)) > convergence_threshold:
            losses_min.append(np.mean(losses_test))
            min_count = 0

        if min_count == convergence_window:
            print(
                f"Convergence detected with last {convergence_window} epochs improvement {losses_min[-1] - np.min(loss_track_test[-convergence_window:])}, ending training..."
            )
            break

    return model, loss_track_train, loss_track_test, params_nonc_names, params_c_names
