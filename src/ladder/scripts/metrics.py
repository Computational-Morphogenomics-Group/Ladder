"""The metrics module houses functions for evaluation.

The functions here are used to assess latent space conditional
mixing / separation and generative accuracy for point clouds.
The generative accuracy metrics are divided into two groups depending
on how they operate. (1) uses traditional regression metrics on pseudobulk
sequence-depth normalized profiles, (2) uses metrics to assess accuracy
of point clouds.
"""

from typing import Literal

import anndata as ad
import numpy as np
import ot
import pyro
import torch
import torch.utils.data as utils
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import trange

####################################
####################################
########### Inner Calls ############
####################################
####################################


# Get tqdm iterator
def _get_iterator(obj, verbose=False):
    if verbose:
        return trange(obj)
    return range(obj)


# _get_normalized_profile implicitly calls this
def _norm_lib_size(x, norm_size=1e4):
    lib_sizes = torch.sum(x, dim=1).add(1)
    return torch.div(x.T, lib_sizes).T * norm_size


# Public version internally calls this
def _get_normalized_profile(point_set, lib_size=1e4):
    return _norm_lib_size(point_set, lib_size).T.mean(-1)


# Get indexes where points belong to target
# _get_subset implicitly calls this
def _get_idxs(point_dataset, target):
    return [
        idx
        for idx in range(len(point_dataset))
        if (point_dataset[idx][1] == target).all()
    ]


# Subset the dataset where target is included
def _get_subset(point_dataset, target):
    tup = point_dataset[_get_idxs(point_dataset, target)]
    return utils.TensorDataset(*tup)


# Compare multiple profiles to a single mean profile - RMSE
def _get_rmse_n_to_1(profiles, mean_profile, **kwargs):
    obj = profiles.add(-1 * mean_profile).square().mean(-1).sqrt()
    return obj.mean().item(), obj.var().item()


# Compare multiple profiles to a single mean profile - Pearson Correlation
def _get_corr_n_to_1(profiles, mean_profile, **kwargs):
    obj = [pearsonr(profile, mean_profile)[0] for profile in profiles]
    return np.mean(obj), np.var(obj)


# Compare multiple point clouds to a single cloud - CD
def _get_chamf_n_to_1(samples, orig, verbose=False, **kwargs):
    matches = [
        torch.cdist(orig, samples[i], p=2).argmin(-1)
        for i in _get_iterator(len(samples), verbose=verbose)
    ]
    chamf = torch.stack(
        [
            orig.add(-1 * (samples[i][matches[i]])).square().mean()
            for i in _get_iterator(len(samples), verbose=verbose)
        ]
    )
    return chamf.mean().item(), chamf.var().item()


# Compare multiple point clouds to a single cloud - 2-SW
def _get_sliced_wasserstein_n_to_1(
    samples, orig, projections=1e3, verbose=False, **kwargs
):
    unif_simplex_a, unif_simplex_b = (
        torch.ones(orig.shape[0]).div(orig.shape[0]),
        torch.ones(samples.shape[1]).div(samples.shape[1]),
    )
    wd = torch.stack(
        [
            ot.sliced_wasserstein_distance(
                orig, samples[i], unif_simplex_a, unif_simplex_b, int(projections)
            )
            for i in _get_iterator(len(samples), verbose=verbose)
        ]
    )
    return wd.mean().item(), wd.var().item()


####################################
####################################
#### Reconstruction - Functions ####
####################################
####################################


def get_normalized_profile(
    point_dataset,
    target: torch.Tensor = None,
    lib_size: float = 1e4,
    batched: bool = False,
):
    """Used to get library-size normalized pseudobulk profile from group of cells.

    The normalized profile can then be compared to other profiles through RMSE
    or Pearson Correlation. Note that we cannot do this with the original data points
    if there is no explicit matching (as in the case of transfer).

    Parameters
    ----------
    point_dataset
        The `TensorDataset` or `ConcatTensorDataset` object to be provided.

    target : torch.Tensor, optional
        Optional target condition encoding tensor to subset the provided `TensorDataset` input.

    lib_size : float, default: 1e4
        Library size to normalize each cell.

    batched : bool, default: False
        If `True`, assumes batch is concatenated to the inputs

    Returns
    -------
    normalized_profile : torch.Tensor
        Normalized pseudo-bulk profile for the given points.
    """
    if target is None:
        point_set = point_dataset[:][0]

    else:
        point_set = _get_subset(point_dataset, target)[:][0]

    if batched:
        point_set = point_set[..., :-1]

    return _norm_lib_size(point_set, lib_size).T.mean(-1)


def gen_profile_reproduction(
    point_dataset,
    model: pyro.infer.Predictive,
    source: torch.Tensor = None,
    target: torch.Tensor = None,
    n_trials: int = 10,
    lib_size: float = 1e4,
    verbose: bool = False,
    use_cuda: bool = False,
):
    """Used to generate new points by performing a full pass through the models.

    The points and their normalized profiles are logged for each generation, up to the
    number of `n_trials`. If `source` and `target` are provided, performs a transfer. Otherwise
    runs for reconstruction.

    Parameters
    ----------
    point_dataset
        The `TensorDataset` or `ConcatTensorDataset` object to be provided.

    model : pyro.infer.Predictive
        The model generator function wrapped in `pyro.infer.Predictive`.

    source : torch.Tensor, optional
        Optional source condition encoding tensor to subset the provided `TensorDataset` input.

    target : torch.Tensor, optional
        Optional target conditon encoding tensor to subset the provided `TensorDataset` input.

    n_trials : int, default: 10
        Number of times to repeat the generative process. Must be positive.

    lib_size : float, default: 1e4
        Library size for normalized profiles.

    verbose : bool, default: False
        If `True`, prints how many iterations are left for `n_trials`.

    use_cuda : bool, default: False
        If `True`, attempts to use CUDA device for the generative process.

    Returns
    -------
    profiles : torch.Tensor
        Normalized pseudo-bulk profiles for the points generated.

    preds : torch.Tensor
        Points generated by the model.
    """
    if source is not None and target is not None:
        source_set, target_set = _get_subset(point_dataset, source), _get_subset(
            point_dataset, target
        )
        _y_target = target_set[0][1].repeat(len(source_set), 1)

    else:
        source_set, target_set = point_dataset, point_dataset
        _y_target = point_dataset[:][1]

    if use_cuda:
        preds = torch.stack(
            [
                model(
                    source_set[:][0].cuda(),
                    y_source=source_set[:][1].cuda(),
                    y_target=_y_target.cuda(),
                )["x"][0].cpu()
                for i in _get_iterator(n_trials, verbose=verbose)
            ]
        )

    else:
        preds = torch.stack(
            [
                model(
                    source_set[:][0].cpu(),
                    y_source=source_set[:][1].cpu(),
                    y_target=_y_target.cpu(),
                )["x"][0].cpu()
                for i in _get_iterator(n_trials, verbose=verbose)
            ]
        )

    profiles = torch.stack([_get_normalized_profile(pred, lib_size) for pred in preds])

    return profiles, preds


def get_reproduction_error(
    point_dataset,
    model: pyro.infer.Predictive,
    source: torch.Tensor = None,
    target: torch.Tensor = None,
    metric: Literal["chamfer", "rmse", "swd", "corr"] = "corr",
    n_trials: int = 10,
    lib_size: float = 1e4,
    batched: bool = False,
    **kwargs,
):
    """Calculates the model generative error for the given metric.

    Wraps around `gen_profile_reproduction` with the `_metric_func`
    corresponding to the chosen metric.

    Parameters
    ----------
    point_dataset
        The `TensorDataset` or `ConcatTensorDataset` object to be provided.

    model : pyro.infer.Predictive
        The model generator function wrapped in `pyro.infer.Predictive`.

    source : torch.Tensor, optional
        Optional source condition encoding tensor to subset the provided `TensorDataset` input.

    target : torch.Tensor, optional
        Optional target conditon encoding tensor to subset the provided `TensorDataset` input.

    metric : {"chamfer", "rmse", "swd", "corr"}, default: "corr"
        Metric to be considered for evaluation.

    n_trials : int, default: 10
        Number of times to repeat the generative process. Must be positive.

    lib_size : float, default: 1e4
        Library size for normalized profiles.

    batched : bool, default: False
        If `True`, assumes batch is concatenated to the inputs.

    **kwargs : dict, optional
        Additional keyword arguments to be passed to `gen_profile_reproduction`.

    Returns
    -------
    preds_mean_error : float
        Mean of the value for the metric calculated across `n_trials` repetitions.

    preds_var_error : float
        Variance of the value for the metric calculated across `n_trials` repetitions.

    pred_profiles : torch.Tensor
        Normalized pseudo-bulk profiles for the points generated.

    preds : torch.Tensor
        Points generated by the model.
    """
    match metric:  # Add different case for each key
        case "rmse":
            _metric_func = _get_rmse_n_to_1

        case "corr":
            _metric_func = _get_corr_n_to_1

        case "chamfer":
            _metric_func = _get_chamf_n_to_1

        case "swd":
            _metric_func = _get_sliced_wasserstein_n_to_1

    pred_profiles, preds = gen_profile_reproduction(
        point_dataset,
        model,
        source,
        target,
        n_trials=n_trials,
        lib_size=lib_size,
        **kwargs,
    )

    match metric:
        case "rmse" | "corr":  # Add profile metrics here
            mean_profile = get_normalized_profile(
                point_dataset, target=target, batched=batched
            )
            preds_mean_error, preds_mean_var = _metric_func(
                pred_profiles, mean_profile, **kwargs
            )

        case "chamfer" | "swd":  # Add cloud metrics here
            if target is None:
                orig = point_dataset[:][0]

            else:
                orig = _get_subset(point_dataset, target)[:][0]

            if batched:
                orig = orig[..., :-1]

            preds_mean_error, preds_var_error = _metric_func(preds, orig, **kwargs)

    return preds_mean_error, preds_var_error, pred_profiles, preds


####################################
####################################
####### MIXING - FUNCTIONS #########
####################################
####################################


def _prep_label_data(anndata: ad.AnnData, test_for: str, embed: str):
    # Grab embeddings and metadata
    X = anndata.obsm[embed]
    y = anndata.obs[test_for]

    # Get encodings for labels
    y_enc = LabelEncoder().fit_transform(list(y))

    return X, y_enc


# KNN Classifier Error with euclidean distance
def knn_error(anndata: ad.AnnData, test_for: str, embed: str, n_neighbors=30):
    """Calculates KNN classifier accuracy on embeddings.

    Trains a KNN classifier on `test_for` using the embeddings `embed`.

    Parameters
    ----------
    anndata : ad.AnnData
        The anndata object provided.

    test_for : str
        Factor column in `anndata.obs` for the original labels.

    embed : str
        Key in `anndata.obsm` pointing to the embeddings to run K-Means with.

    n_neighbors : int, default: 30
        K for the KNN classifier.

    Returns
    -------
    float
        Accuracy of the KNN classifier.
    """
    # Prep data
    X, y = _prep_label_data(anndata, test_for, embed)

    # Run kNN
    model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)

    return model.score(X, y)


# Run k-means for nmi / ari
def _run_k_means(anndata: ad.AnnData, test_for: str, embed: str):
    # Prep data
    X, y = _prep_label_data(anndata, test_for, embed)

    # Run k-Means
    y_pred = KMeans(
        n_clusters=len(np.unique(y)),
        init="k-means++",
        n_init="auto",
        max_iter=300,
        random_state=42,
    ).fit_predict(X)

    return y, y_pred


def kmeans_nmi(anndata: ad.AnnData, test_for: str, embed: str):
    """Calculates the K-Means NMI for latent representations.

    The K-Means clustering in the latents is compared with the labels
    originally appearing in `anndata`.

    Parameters
    ----------
    anndata : ad.AnnData
        The anndata object provided.

    test_for : str
        Factor column in `anndata.obs` for the original labels.

    embed : str
        Key in `anndata.obsm` pointing to the embeddings to run K-Means with.

    Returns
    -------
    float
        NMI for original labels and K-Means clusters.
    """
    y, y_pred = _run_k_means(anndata, test_for, embed)

    return normalized_mutual_info_score(y, y_pred)


def kmeans_ari(anndata: ad.AnnData, test_for: str, embed: str):
    """Calculates the K-Means ARI for latent representations.

    The K-Means clustering in the latents is compared with the labels
    originally appearing in `anndata`.

    Parameters
    ----------
    anndata : ad.AnnData
        The anndata object provided.

    test_for : str
        Factor column in `anndata.obs` for the original labels.

    embed : str
        Key in `anndata.obsm` pointing to the embeddings to run K-Means with.

    Returns
    -------
    float
        ARI for original labels and K-Means clusters.
    """
    y, y_pred = _run_k_means(anndata, test_for, embed)

    return adjusted_rand_score(y, y_pred)


def calc_asw(anndata: ad.AnnData, test_for: str, embed: str):
    """Calculates the normalized Average Silhouette Width (ASW)

    Courtesy of scib (2021, 10.1038/s41592-021-01336-8)
    Source: https://github.com/theislab/scib/blob/main/scib/metrics/silhouette.py

    Parameters
    ----------
    anndata : ad.AnnData
        The anndata object provided.

    test_for : str
        Factor column in `anndata.obs` for the original labels.

    embed : str
        Key in `anndata.obsm` pointing to the embeddings to run K-Means with.

    Returns
    -------
    float
        Average Silhouette Width normalized to the range [0,1].
    """
    # Prep data
    X, y = _prep_label_data(anndata, test_for, embed)

    asw = silhouette_score(X=X, labels=y)

    # Normalize width
    asw = (asw + 1) / 2
    return asw
