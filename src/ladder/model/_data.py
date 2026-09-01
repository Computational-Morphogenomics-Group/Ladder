from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)


def validate_registered_anndata(adata: AnnData, registry: dict):
    """Validate the registered feature names, feature order, and data layer."""
    if adata.n_vars != registry["n_vars"]:
        raise ValueError(
            f"AnnData has {adata.n_vars} variables, expected {registry['n_vars']}."
        )

    if list(map(str, adata.var_names)) != registry["var_names"]:
        raise ValueError(
            "AnnData var_names do not match the features and order registered by setup_anndata."
        )

    layer = registry["layer"]
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"AnnData is missing registered layer: {layer!r}.")


def get_registered_matrix(adata: AnnData, registry: dict):
    """Return the data matrix selected during setup_anndata."""
    layer = registry["layer"]
    return adata.layers[layer] if layer is not None else adata.X


def _encode_obs_categories(
    adata: AnnData, key: str, categories: list[str]
) -> np.ndarray:
    if key not in adata.obs:
        raise ValueError(f"AnnData is missing required obs column: {key!r}.")

    category_to_index = {category: i for i, category in enumerate(categories)}
    values = adata.obs[key].astype(str)

    try:
        indices = np.array(
            [category_to_index[value] for value in values], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(
            f"AnnData obs column {key!r} contains category {error.args[0]!r}, "
            "which was not present during setup_anndata."
        ) from error

    one_hot = np.zeros((adata.n_obs, len(categories)), dtype=np.float32)
    one_hot[np.arange(adata.n_obs), indices] = 1.0
    return one_hot


def encode_registered_conditions(adata: AnnData, registry: dict) -> np.ndarray:
    """Return concatenated one-hot encodings for the registered condition columns."""
    encoded = [
        _encode_obs_categories(adata, key, registry["condition_categories"][key])
        for key in registry["condition_keys"]
    ]
    return np.concatenate(encoded, axis=1)


def encode_registered_batches(adata: AnnData, registry: dict) -> np.ndarray | None:
    """One-hot encode the registered batch column, if present."""
    batch_key = registry["batch_key"]
    if batch_key is None:
        return None
    return _encode_obs_categories(adata, batch_key, registry["batch_categories"])


def normalize_indices(indices, n_obs: int) -> np.ndarray:
    """Convert an observation selection to a one-dimensional array of integer positions."""
    if indices is None:
        normalized = np.arange(n_obs, dtype=np.int64)
    else:
        normalized = np.asarray(indices)
        if normalized.ndim != 1:
            raise ValueError("indices must be one-dimensional.")
        if normalized.dtype == bool:
            if normalized.shape[0] != n_obs:
                raise ValueError("Boolean indices must have length adata.n_obs.")
            normalized = np.flatnonzero(normalized)
        elif normalized.dtype.kind not in "iu":
            raise TypeError(
                "indices must contain integer observation positions or a boolean mask."
            )
        normalized = normalized.astype(np.int64, copy=False)

    if normalized.size == 0:
        raise ValueError("indices must select at least one observation.")
    if np.any(normalized < 0) or np.any(normalized >= n_obs):
        raise ValueError(
            "indices contain positions outside the AnnData observation range."
        )
    return normalized


def get_condition_groups(adata: AnnData, registry: dict) -> np.ndarray:
    """Return joint condition labels used to stratify generated data splits."""
    condition_keys = registry["condition_keys"]
    if len(condition_keys) == 1:
        return adata.obs[condition_keys[0]].astype(str).to_numpy()
    return adata.obs[condition_keys].astype(str).agg("||".join, axis=1).to_numpy()


def stratified_train_validation_indices(
    groups: np.ndarray,
    *,
    train_size: float,
    split_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every condition group and assign the remainder to validation."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be in the open interval (0, 1).")

    rng = np.random.default_rng(split_seed)
    train_indices = []
    validation_indices = []

    for group in np.unique(groups):
        group_indices = np.flatnonzero(groups == group)
        if len(group_indices) < 2:
            raise ValueError(
                f"Condition group {group!r} has fewer than 2 observations and cannot be split."
            )

        group_indices = rng.permutation(group_indices)
        n_train = max(1, int(np.floor(len(group_indices) * train_size)))
        train_indices.append(group_indices[:n_train])
        validation_indices.append(group_indices[n_train:])

    return (
        np.sort(np.concatenate(train_indices)).astype(np.int64),
        np.sort(np.concatenate(validation_indices)).astype(np.int64),
    )


def validate_train_validation_indices(
    train_indices,
    validation_indices,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate explicitly supplied training and validation positions."""

    def _validate(indices, name):
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if indices.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if indices.dtype.kind not in "iu":
            raise TypeError(f"{name} must contain integer observation positions.")

        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= n_obs):
            raise ValueError(
                f"{name} contains positions outside the AnnData observation range."
            )
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} contains duplicate positions.")
        return indices

    train_indices = _validate(train_indices, "train_indices")
    validation_indices = _validate(validation_indices, "validation_indices")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("train_indices and validation_indices must not overlap.")
    return train_indices, validation_indices


def as_numpy_array(matrix) -> np.ndarray:
    """Convert a dense or sparse matrix to a float32 NumPy array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


class _AnnDataBatchDataset(Dataset):
    """Dataset that loads and converts complete minibatches."""

    def __init__(self, matrix, indices, labels, batch_covariates=None):
        self.matrix = matrix
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.batch_covariates = (
            None
            if batch_covariates is None
            else np.asarray(batch_covariates, dtype=np.float32)
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, items):
        obs_indices = self.indices[np.asarray(items, dtype=np.int64)]
        # Backed HDF5 matrices require ordered indices for efficient slicing.
        obs_indices = np.sort(obs_indices)
        result = [
            torch.from_numpy(as_numpy_array(self.matrix[obs_indices])),
            torch.from_numpy(self.labels[obs_indices]),
        ]
        if self.batch_covariates is not None:
            result.append(torch.from_numpy(self.batch_covariates[obs_indices]))
        return tuple(result)


def make_data_loader(
    matrix,
    labels,
    indices,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    batch_covariates=None,
) -> DataLoader:
    """Load each minibatch with one matrix access before converting to tensors."""
    dataset = _AnnDataBatchDataset(
        matrix,
        indices,
        labels,
        batch_covariates=batch_covariates,
    )
    index_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    batch_sampler = BatchSampler(
        index_sampler, batch_size=batch_size, drop_last=drop_last
    )
    return DataLoader(
        dataset,
        # The sampler supplies complete batches for the dataset to load at once.
        sampler=batch_sampler,
        batch_size=None,
        pin_memory=torch.cuda.is_available(),
    )
