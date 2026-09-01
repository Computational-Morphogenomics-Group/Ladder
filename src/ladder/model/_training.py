from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import Trace_ELBO

from ._data import (
    encode_registered_batches,
    encode_registered_conditions,
    get_condition_groups,
    get_registered_matrix,
    make_data_loader,
    stratified_train_validation_indices,
    validate_registered_anndata,
    validate_train_validation_indices,
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _AdversarialPyroTrainer:
    """Small alternating trainer for DLVAE-style adversarial Pyro modules."""

    def __init__(
        self,
        module,
        train_loader,
        validation_loader,
        *,
        max_epochs: int = 400,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        model_steps: int = 1,
        classifier_steps: int = 1,
        early_stopping: bool = False,
        patience: int = 20,
        min_delta: float = 0.0,
        kl_warmup_epochs: int = 20,
        device=None,
    ):
        self.module = module.to(_resolve_device(device))
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.max_epochs = max_epochs
        self.model_steps = model_steps
        self.classifier_steps = classifier_steps
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.target_z_kl_weight = float(module.z_kl_weight)
        self.target_w_kl_weight = float(module.w_kl_weight)
        self.elbo = Trace_ELBO()
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "validation_loss": [],
            "train_classifier_loss": [],
            "validation_classifier_loss": [],
            "z_kl_weight": [],
            "w_kl_weight": [],
            "kl_warmup_fraction": [],
        }

        self.model_params, self.classifier_params = self._split_parameters()
        self.model_optimizer = torch.optim.AdamW(
            self.model_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.classifier_optimizer = torch.optim.AdamW(
            self.classifier_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def train(self) -> dict[str, list[float]]:
        try:
            return self._train()
        finally:
            self._set_target_kl_weights()
            self._set_requires_grad(self.module.parameters(), True)
            self.module.eval()

    def _train(self) -> dict[str, list[float]]:
        best_loss = np.inf
        best_state = None
        final_loss = np.inf
        epochs_without_improvement = 0
        print(f"Training {self.module.__class__.__name__} on {self.device}")

        for epoch in range(self.max_epochs):
            warmup_fraction = self._set_epoch_kl_weights(epoch)
            checkpointing_active = warmup_fraction >= 1.0
            should_stop = False

            train_loss, train_classifier_loss = self._train_epoch()

            (
                validation_loss,
                validation_classifier_loss,
                validation_components,
            ) = self._validate_epoch()
            metrics = {
                "training loss": train_loss,
                "training classifier loss": train_classifier_loss,
                "validation loss": validation_loss,
                "validation classifier loss": validation_classifier_loss,
                **validation_components,
            }
            for name, value in metrics.items():
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite {name} at epoch {epoch + 1}: {value}."
                    )
            validation_reconstruction_w_nll = validation_components.get(
                "reconstruction_w_nll", np.nan
            )
            validation_reconstruction_z_nll = validation_components.get(
                "reconstruction_z_nll", np.nan
            )

            self.history["train_loss"].append(train_loss)
            self.history["validation_loss"].append(validation_loss)
            self.history["train_classifier_loss"].append(train_classifier_loss)
            self.history["validation_classifier_loss"].append(
                validation_classifier_loss
            )
            self.history["z_kl_weight"].append(float(self.module.z_kl_weight))
            self.history["w_kl_weight"].append(float(self.module.w_kl_weight))
            self.history["kl_warmup_fraction"].append(warmup_fraction)
            for component, value in validation_components.items():
                self.history.setdefault(f"validation_{component}", []).append(value)

            final_loss = validation_loss

            if checkpointing_active:
                if validation_loss < best_loss - self.min_delta:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(self.module.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    should_stop = (
                        self.early_stopping
                        and epochs_without_improvement >= self.patience
                    )

            print(
                f"Epoch {epoch + 1} | "
                f"kl_warmup={warmup_fraction:.3f} | "
                f"train_loss={train_loss:.4f} | "
                f"validation_loss={validation_loss:.4f} | "
                f"validation_reconstruction_w_nll={validation_reconstruction_w_nll:.4f} | "
                f"validation_reconstruction_z_nll={validation_reconstruction_z_nll:.4f} | "
                f"train_classifier_loss={train_classifier_loss:.4f} | "
                f"validation_classifier_loss={validation_classifier_loss:.4f} | "
                f"best_validation_loss={best_loss:.4f} | "
                f"patience={epochs_without_improvement}/{self.patience}"
            )

            if should_stop:
                break

        if best_state is None:
            best_loss = final_loss
            best_state = copy.deepcopy(self.module.state_dict())

        self.best_loss = best_loss
        self.module.load_state_dict(best_state)
        print(
            f"Finished training | epochs={len(self.history['train_loss'])} | "
            f"best_validation_loss={best_loss:.4f}"
        )
        return self.history

    def _set_epoch_kl_weights(self, epoch: int) -> float:
        if self.kl_warmup_epochs == 0:
            warmup_fraction = 1.0
        else:
            warmup_fraction = min(float(epoch + 1) / float(self.kl_warmup_epochs), 1.0)

        self.module.z_kl_weight = self.target_z_kl_weight * warmup_fraction
        self.module.w_kl_weight = self.target_w_kl_weight * warmup_fraction
        return warmup_fraction

    def _set_target_kl_weights(self):
        self.module.z_kl_weight = self.target_z_kl_weight
        self.module.w_kl_weight = self.target_w_kl_weight

    def _split_parameters(self):
        classifier_params = list(self.module.classifiers.parameters())
        classifier_ids = {id(parameter) for parameter in classifier_params}
        model_params = [
            parameter
            for parameter in self.module.parameters()
            if id(parameter) not in classifier_ids
        ]

        if not model_params:
            raise ValueError("No model parameters found.")
        if not classifier_params:
            raise ValueError("No classifier parameters found.")

        return model_params, classifier_params

    def _train_epoch(self):
        model_losses = []
        classifier_losses = []

        for batch in self.train_loader:
            x, y, batch_covariates = self._send_batch_to_device(batch)

            for _ in range(self.classifier_steps):
                classifier_losses.append(self._classifier_step(x, y, batch_covariates))

            for _ in range(self.model_steps):
                model_losses.append(self._model_step(x, y, batch_covariates))

        return float(np.mean(model_losses)), float(np.mean(classifier_losses))

    def _validate_epoch(self):
        self.module.eval()
        model_loss_total = 0.0
        classifier_loss_total = 0.0
        component_totals = {}
        component_n = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                x, y, batch_covariates = self._send_batch_to_device(batch)
                model_loss = self.elbo.loss(
                    self.module.model,
                    self.module.guide,
                    x,
                    y,
                    batch_covariates,
                )
                classifier_loss = (
                    self.module.classifier_loss(x, y, batch_covariates)
                    .detach()
                    .cpu()
                    .item()
                )
                batch_n = int(x.shape[0])
                model_loss_total += model_loss * batch_n
                classifier_loss_total += classifier_loss * batch_n
                components = self.module.loss_components(x, y, batch_covariates)
                for key, value in components.items():
                    component_totals[key] = component_totals.get(key, 0.0) + (
                        float(value.detach().cpu()) * batch_n
                    )
                component_n += batch_n

        component_means = {
            key: value / component_n for key, value in component_totals.items()
        }
        return (
            model_loss_total / component_n,
            classifier_loss_total / component_n,
            component_means,
        )

    @staticmethod
    def _set_requires_grad(parameters, requires_grad: bool):
        for parameter in parameters:
            parameter.requires_grad_(requires_grad)

    def _set_training_phase(self, *, classifier_phase: bool):
        self.module.train(not classifier_phase)
        self.module.classifiers.train(classifier_phase)
        self._set_requires_grad(self.model_params, not classifier_phase)
        self._set_requires_grad(self.classifier_params, classifier_phase)

    def _classifier_step(self, x, y, batch_covariates=None) -> float:
        self._set_training_phase(classifier_phase=True)
        self.module.zero_grad(set_to_none=True)
        loss = self.module.classifier_loss(x, y, batch_covariates)
        return self._optimizer_step(
            loss,
            self.classifier_params,
            self.classifier_optimizer,
            "classifier",
        )

    def _model_step(self, x, y, batch_covariates=None) -> float:
        self._set_training_phase(classifier_phase=False)
        self.module.zero_grad(set_to_none=True)
        loss = self.elbo.differentiable_loss(
            self.module.model,
            self.module.guide,
            x,
            y,
            batch_covariates,
        )
        return self._optimizer_step(
            loss, self.model_params, self.model_optimizer, "model"
        )

    @staticmethod
    def _optimizer_step(loss, parameters, optimizer, phase: str) -> float:
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {phase} loss; optimizer step was not applied."
            )
        loss.backward(inputs=parameters)
        # An infinite limit checks gradient finiteness without clipping finite gradients.
        torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=float("inf"), error_if_nonfinite=True
        )
        optimizer.step()
        return loss.detach().cpu().item()

    def _send_batch_to_device(self, batch):
        x, y, *batch_covariates = batch
        batch_covariates = (
            None if not batch_covariates else batch_covariates[0].to(self.device)
        )
        return x.to(self.device), y.to(self.device), batch_covariates


def train_discover(
    model,
    *,
    max_epochs: int,
    batch_size: int,
    train_indices,
    validation_indices,
    train_size: float,
    split_seed: int | None,
    random_seed: int | None,
    lr: float,
    weight_decay: float,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    kl_warmup_epochs: int,
    model_steps: int,
    classifier_steps: int,
    device=None,
    output_dir=None,
) -> dict[str, list[float]]:
    """Train a Discover model and record its fitted state and artifacts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if random_seed is not None:
        pyro.util.set_rng_seed(random_seed)

    validate_registered_anndata(model.adata, model.adata_registry)
    if not model.adata.obs_names.is_unique:
        raise ValueError(
            "AnnData obs_names must be unique to save reusable split indices."
        )

    x = get_registered_matrix(model.adata, model.adata_registry)
    y = encode_registered_conditions(model.adata, model.adata_registry)
    batch_covariates = encode_registered_batches(model.adata, model.adata_registry)

    indices_provided = train_indices is not None or validation_indices is not None
    if indices_provided:
        if train_indices is None or validation_indices is None:
            raise ValueError(
                "train_indices and validation_indices must be provided together."
            )
        train_indices, validation_indices = validate_train_validation_indices(
            train_indices,
            validation_indices,
            model.adata.n_obs,
        )
        effective_split_seed = None
    else:
        effective_split_seed = split_seed
        train_indices, validation_indices = stratified_train_validation_indices(
            get_condition_groups(model.adata, model.adata_registry),
            train_size=train_size,
            split_seed=effective_split_seed,
        )
    train_loader = make_data_loader(
        x,
        y,
        train_indices,
        batch_size=batch_size,
        shuffle=True,
        # BatchNorm cannot estimate batch statistics from a single observation.
        drop_last=(
            model.module.use_batch_norm and len(train_indices) % batch_size == 1
        ),
        batch_covariates=batch_covariates,
    )
    validation_loader = make_data_loader(
        x,
        y,
        validation_indices,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        batch_covariates=batch_covariates,
    )

    trainer = _AdversarialPyroTrainer(
        model.module,
        train_loader,
        validation_loader,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        model_steps=model_steps,
        classifier_steps=classifier_steps,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        kl_warmup_epochs=kl_warmup_epochs,
        device=device,
    )
    model.is_trained = False
    history = trainer.train()

    model.history = history
    model.module = trainer.module
    model.device = trainer.device
    model.train_indices = train_indices
    model.validation_indices = validation_indices
    model.training_obs_names = np.asarray(model.adata.obs_names, dtype=str)
    model.best_validation_loss = trainer.best_loss
    model.training_params = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "train_size": None if indices_provided else train_size,
        "split_seed": effective_split_seed,
        "random_seed": random_seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "early_stopping": early_stopping,
        "patience": patience,
        "min_delta": min_delta,
        "kl_warmup_epochs": kl_warmup_epochs,
        "model_steps": model_steps,
        "classifier_steps": classifier_steps,
        "device": str(model.device),
    }
    model.is_trained = True

    if output_dir is not None:
        model.save(output_dir)
        _write_training_artifacts(model, output_dir)

    return history


def _training_config(model) -> dict:
    return {
        "model_class": model.__class__.__name__,
        "module_class": model.module.__class__.__name__,
        "adata_registry": model.adata_registry,
        "init_params": model.init_params,
        "training_params": getattr(model, "training_params", None),
        "n_obs": None if model.adata is None else int(model.adata.n_obs),
        "n_vars": model.adata_registry["n_vars"],
        "n_train": (
            None
            if getattr(model, "train_indices", None) is None
            else int(len(model.train_indices))
        ),
        "n_validation": (
            None
            if getattr(model, "validation_indices", None) is None
            else int(len(model.validation_indices))
        ),
        "best_validation_loss": getattr(model, "best_validation_loss", None),
        "artifacts": {
            "model": "model.pt",
            "training_config": "training_config.json",
            "history_csv": "history.csv",
            "log": "log.txt",
        },
    }


def _write_training_artifacts(model, path: str | Path):
    history = getattr(model, "history", None)
    if history is None:
        return

    path = Path(path)
    with (path / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(_training_config(model), handle, indent=2, default=_json_default)
        handle.write("\n")

    _write_history_csv(history, path / "history.csv")

    with (path / "log.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"model\t{model.__class__.__name__}\n")
        handle.write(f"module\t{model.module.__class__.__name__}\n")
        handle.write(f"device\t{getattr(model, 'device', None)}\n")
        handle.write(f"n_train\t{len(model.train_indices)}\n")
        handle.write(f"n_validation\t{len(model.validation_indices)}\n")
        handle.write(f"best_validation_loss\t{model.best_validation_loss:.6g}\n")
        handle.write(f"epochs\t{len(history['train_loss'])}\n")
        handle.write("history_csv\thistory.csv\n")


def _write_history_csv(history: dict[str, list[float]], path: Path):
    columns = list(history)
    n_epochs = max((len(values) for values in history.values()), default=0)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *columns])
        writer.writeheader()
        for epoch in range(n_epochs):
            row = {"epoch": epoch + 1}
            for column in columns:
                values = history[column]
                row[column] = values[epoch] if epoch < len(values) else ""
            writer.writerow(row)
