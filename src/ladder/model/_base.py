from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch import Tensor, nn

from ._data import (
    as_numpy_array,
    encode_registered_batches,
    encode_registered_conditions,
    get_registered_matrix,
    normalize_indices,
    validate_registered_anndata,
)
from ._training import train_discover

AdversarialInput = Literal["reconstruction", "z"]
DecoderType = Literal["nonlinear", "linear"]
LoadingDecoder = Literal["z", "w_z"]
LatentBlock = Literal["z", "w"]

_Discover = TypeVar("_Discover", bound="BaseDiscover")


def _as_list(
    value: str | Sequence[str] | None,
    *,
    field_name: str,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError as error:
        raise TypeError(
            f"{field_name} must be a string or sequence of strings."
        ) from error


def _categories_from_obs(adata: AnnData, key: str) -> list[str]:
    if key not in adata.obs:
        raise ValueError(f"AnnData is missing required obs column: {key!r}.")

    values = adata.obs[key]
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = list(values.cat.categories)
    else:
        categories = list(pd.Index(values.astype(str)).unique())

    if len(categories) == 0:
        raise ValueError(f"AnnData obs column {key!r} has no categories.")
    return [str(category) for category in categories]


def _state_dict_to_cpu(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


class BaseDiscover:
    """Shared AnnData-aware lifecycle for Discover models."""

    _registry_key: ClassVar[str]
    _likelihood_param: ClassVar[str]
    _module_classes: ClassVar[dict[str, type[nn.Module]]]

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        *,
        condition_keys: str | Sequence[str],
        layer: str | None = None,
        batch_key: str | None = None,
    ):
        """Register the AnnData fields used for training and inference.

        The registration records the ordered features and categorical encodings
        in ``adata.uns``. The likelihood is evaluated against the untransformed
        matrix selected by ``layer``.
        """
        condition_keys = _as_list(condition_keys, field_name="condition_keys")
        if len(condition_keys) == 0:
            raise ValueError(f"{cls.__name__} requires at least one condition key.")

        if layer is not None and layer not in adata.layers:
            raise ValueError(f"AnnData is missing requested layer: {layer!r}.")

        condition_categories = {
            key: _categories_from_obs(adata, key) for key in condition_keys
        }
        batch_categories = None
        if batch_key is not None:
            batch_categories = _categories_from_obs(adata, batch_key)

        adata.uns[cls._registry_key] = {
            "layer": layer,
            "n_vars": int(adata.n_vars),
            "var_names": list(map(str, adata.var_names)),
            "condition_keys": condition_keys,
            "condition_categories": condition_categories,
            "label_dims": [len(condition_categories[key]) for key in condition_keys],
            "batch_key": batch_key,
            "batch_categories": batch_categories,
        }

    def _initialize(
        self,
        adata: AnnData,
        *,
        likelihood: str,
        init_params: dict[str, Any],
    ):
        if self._registry_key not in adata.uns:
            raise ValueError(
                f"Call {self.__class__.__name__}.setup_anndata(adata, ...) before initializing the model."
            )

        self._module_cls = self._get_module_cls(likelihood)
        self.adata = adata
        self.adata_registry = copy.deepcopy(adata.uns[self._registry_key])

        self.init_params = copy.deepcopy(init_params)
        self.module_init_params = self._get_module_init_params()
        self.module = self._module_cls(**self.module_init_params)
        self.is_trained = False

    def train(
        self,
        *,
        max_epochs: int = 400,
        batch_size: int = 256,
        train_indices: Sequence[int] | np.ndarray | None = None,
        validation_indices: Sequence[int] | np.ndarray | None = None,
        train_size: float = 0.9,
        split_seed: int | None = 0,
        random_seed: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping: bool = True,
        patience: int = 20,
        min_delta: float = 0.0,
        kl_warmup_epochs: int = 0,
        model_steps: int = 1,
        classifier_steps: int = 1,
        device: str | torch.device | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Train using provided indices or a condition-stratified split.

        Generated splits assign every observation to training or validation.
        ``split_seed`` controls the split; ``random_seed`` controls the rest of training.
        The best validation checkpoint after KL warmup is restored. ``early_stopping``
        controls whether training may finish before ``max_epochs``.
        """
        return train_discover(
            self,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_indices=train_indices,
            validation_indices=validation_indices,
            train_size=train_size,
            split_seed=split_seed,
            random_seed=random_seed,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            kl_warmup_epochs=kl_warmup_epochs,
            model_steps=model_steps,
            classifier_steps=classifier_steps,
            device=device,
            output_dir=output_dir,
        )

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        *,
        indices: Sequence[int] | Sequence[bool] | np.ndarray | None = None,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        """Return deterministic posterior means for z and w latents."""
        adata = self.adata if adata is None else adata
        if adata is None:
            raise ValueError(
                "An AnnData object is required to compute latent representations."
            )

        validate_registered_anndata(adata, self.adata_registry)
        x_matrix = get_registered_matrix(adata, self.adata_registry)
        y = encode_registered_conditions(adata, self.adata_registry)
        indices = normalize_indices(indices, adata.n_obs)

        device = next(self.module.parameters()).device
        self.module.eval()

        z_chunks = []
        w_chunks = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                x_batch = torch.from_numpy(as_numpy_array(x_matrix[batch_indices])).to(
                    device
                )
                y_batch = torch.from_numpy(y[batch_indices]).to(device)
                outputs = self.module.inference(x_batch, y_batch)
                z_chunks.append(outputs["z_loc"].detach().cpu().numpy())
                w_chunks.append(outputs["w_loc"].detach().cpu().numpy())

        return {
            "z": np.concatenate(z_chunks, axis=0),
            "w": np.concatenate(w_chunks, axis=0),
        }

    def get_loadings(
        self,
        *,
        decoder: LoadingDecoder = "z",
        latent: LatentBlock = "z",
    ) -> pd.DataFrame:
        """Return feature-by-latent coefficients from a linear decoder.

        ``decoder`` selects the Z-only or W-and-Z decoder. ``latent`` selects
        coefficients for Z or W; the Z-only decoder has no W coefficients.
        Batch-covariate columns are excluded.

        Gaussian coefficients act directly on the reconstruction mean. NB and
        Poisson coefficients act on pre-softmax logits and are centered across
        features to remove the shared-logit offset ambiguity.
        """
        if self.module.decoder_type != "linear":
            raise ValueError(
                'Loadings require a model initialized with decoder_type="linear".'
            )
        if decoder not in {"z", "w_z"}:
            raise ValueError("decoder must be 'z' or 'w_z'.")
        if latent not in {"w", "z"}:
            raise ValueError("latent must be 'w' or 'z'.")
        if decoder == "z" and latent != "z":
            raise ValueError("The Z-only decoder has no W loading block.")

        decoder_module = (
            self.module.decoder if decoder == "w_z" else self.module.decoder_z
        )
        start = self.module.w_dim if decoder == "w_z" and latent == "z" else 0
        width = self.module.w_dim if latent == "w" else self.module.latent_dim
        weights = (
            decoder_module.layers[0]
            .weight[:, start : start + width]
            .detach()
            .cpu()
            .clone()
        )
        likelihood = self.init_params[self._likelihood_param]
        if likelihood in {"nb", "poisson"}:
            weights -= weights.mean(dim=0, keepdim=True)
        return pd.DataFrame(
            weights.numpy(),
            index=pd.Index(self.adata_registry["var_names"], name="feature"),
            columns=[f"{latent}_{index}" for index in range(width)],
        )

    def save(self, path: str | Path):
        """Save model weights, configuration, history, and data registration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_class": self.__class__.__name__,
            "adata_registry": self.adata_registry,
            "init_params": self.init_params,
            "module_init_params": self.module_init_params,
            "module_state_dict": _state_dict_to_cpu(self.module.state_dict()),
            "history": getattr(self, "history", None),
            "is_trained": self.is_trained,
            "train_indices": getattr(self, "train_indices", None),
            "validation_indices": getattr(self, "validation_indices", None),
            "training_obs_names": getattr(self, "training_obs_names", None),
            "best_validation_loss": getattr(self, "best_validation_loss", None),
            "training_params": getattr(self, "training_params", None),
        }
        torch.save(payload, path / "model.pt")

    @classmethod
    def load(
        cls: type[_Discover],
        path: str | Path,
        adata: AnnData | None = None,
        map_location: str | torch.device | None = None,
    ) -> _Discover:
        """Load model weights with an optional compatible AnnData object.

        The AnnData object may contain different cells, but must retain the
        registered features and feature order. Its categorical values must be
        among those registered during training. Saved split indices continue
        to refer to ``training_obs_names``. Optimizer state is not restored.
        """
        path = Path(path)
        payload = torch.load(
            path / "model.pt", map_location=map_location, weights_only=False
        )
        model_class = payload["model_class"]
        if model_class != cls.__name__:
            raise ValueError(
                f"Checkpoint contains {model_class}, but {cls.__name__}.load() was called."
            )

        obj = cls.__new__(cls)
        obj.adata = adata
        obj.adata_registry = copy.deepcopy(payload["adata_registry"])
        obj.init_params = copy.deepcopy(payload["init_params"])
        obj._module_cls = cls._get_module_cls(obj.init_params[cls._likelihood_param])
        obj.module_init_params = copy.deepcopy(payload["module_init_params"])
        obj.module = obj._module_cls(**obj.module_init_params)
        obj.module.load_state_dict(payload["module_state_dict"])

        if map_location is not None:
            obj.module.to(torch.device(map_location))

        training_obs_names = payload["training_obs_names"]
        if training_obs_names is not None:
            training_obs_names = np.asarray(training_obs_names, dtype=str)

        if adata is not None:
            validate_registered_anndata(adata, obj.adata_registry)
            encode_registered_conditions(adata, obj.adata_registry)
            encode_registered_batches(adata, obj.adata_registry)
            adata.uns[cls._registry_key] = copy.deepcopy(obj.adata_registry)

        obj.history = payload["history"]
        obj.train_indices = payload["train_indices"]
        obj.validation_indices = payload["validation_indices"]
        obj.training_obs_names = training_obs_names
        obj.best_validation_loss = payload["best_validation_loss"]
        obj.training_params = payload["training_params"]
        obj.is_trained = payload["is_trained"]
        obj.module.eval()

        return obj

    @classmethod
    def _get_module_cls(cls, likelihood: str) -> type[nn.Module]:
        if likelihood not in cls._module_classes:
            valid = ", ".join(sorted(cls._module_classes))
            raise ValueError(f"{cls._likelihood_param} must be one of: {valid}.")
        return cls._module_classes[likelihood]

    def _get_covariate_dim(self) -> int:
        batch_categories = self.adata_registry["batch_categories"]
        if batch_categories is None:
            return 0
        return len(batch_categories)

    def _get_module_init_params(self) -> dict[str, Any]:
        module_init_params = {
            "in_dim": self.adata_registry["n_vars"],
            "label_dims": self.adata_registry["label_dims"],
            "hidden_dim": self.init_params["hidden_dim"],
            "num_layers": self.init_params["num_layers"],
            "use_batch_norm": self.init_params["use_batch_norm"],
            "dropout_rate": self.init_params["dropout_rate"],
            "decoder_type": self.init_params["decoder_type"],
            "latent_dim": self.init_params["latent_dim"],
            "w_dim": self.init_params["w_dim"],
            "recon_weight": self.init_params["recon_weight"],
            "recon_weight_z": self.init_params["recon_weight_z"],
            "z_kl_weight": self.init_params["z_kl_weight"],
            "w_kl_weight": self.init_params["w_kl_weight"],
            "adversarial_weight": self.init_params["adversarial_weight"],
            "classifier_layers": self.init_params["classifier_layers"],
            "learnable_prior": self.init_params["learnable_prior"],
            "covariate_dim": self._get_covariate_dim(),
            "log_variational": self.init_params["log_variational"],
            "adversarial_input": self.init_params["adversarial_input"],
        }
        return module_init_params

    @torch.inference_mode()
    def _decode_posterior(
        self,
        adata: AnnData | None = None,
        *,
        indices: Sequence[int] | Sequence[bool] | np.ndarray | None = None,
        z_only: bool = False,
        use_posterior_mean: bool = False,
        n_samples: int = 1,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode observations, decode posterior values, and average sampled draws."""
        adata = self.adata if adata is None else adata
        if adata is None:
            raise ValueError("An AnnData object is required to compute decoded output.")
        validate_registered_anndata(adata, self.adata_registry)
        x_matrix = get_registered_matrix(adata, self.adata_registry)
        y = encode_registered_conditions(adata, self.adata_registry)
        batch_covariates = encode_registered_batches(adata, self.adata_registry)
        indices = normalize_indices(indices, adata.n_obs)

        effective_n_samples = 1 if use_posterior_mean else n_samples

        device = next(self.module.parameters()).device
        self.module.eval()
        output_chunks = []

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            x_batch = torch.from_numpy(as_numpy_array(x_matrix[batch_indices])).to(
                device
            )
            y_batch = torch.from_numpy(y[batch_indices]).to(device)
            batch_covariates_batch = (
                None
                if batch_covariates is None
                else torch.from_numpy(batch_covariates[batch_indices]).to(device)
            )

            inference_outputs = self.module.inference(x_batch, y_batch)
            if use_posterior_mean:
                z = inference_outputs["z_loc"].unsqueeze(0)
            else:
                z = torch.distributions.Normal(
                    inference_outputs["z_loc"],
                    inference_outputs["z_scale"],
                ).sample((effective_n_samples,))

            sample_shape = z.shape[:2]
            z = z.reshape(-1, z.shape[-1])
            decoder_batch_covariates = None
            if batch_covariates_batch is not None:
                decoder_batch_covariates = (
                    batch_covariates_batch.unsqueeze(0)
                    .expand(effective_n_samples, -1, -1)
                    .reshape(-1, batch_covariates_batch.shape[-1])
                )

            w = None
            if not z_only:
                if use_posterior_mean:
                    w = inference_outputs["w_loc"].unsqueeze(0)
                else:
                    w = torch.distributions.Normal(
                        inference_outputs["w_loc"],
                        inference_outputs["w_scale"],
                    ).sample((effective_n_samples,))
                w = w.reshape(-1, w.shape[-1])

            output = self.module.decode(
                z,
                w,
                decoder_batch_covariates,
            )
            output = output.reshape(*sample_shape, adata.n_vars)
            output_chunks.append(output.cpu().numpy())

        output = np.concatenate(output_chunks, axis=1)
        return output.mean(axis=0)
