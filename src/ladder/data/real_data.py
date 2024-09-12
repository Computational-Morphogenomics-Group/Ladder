"""The real_data module houses the functions to be used with real datasets.

The functions defined here can either be used independently for
specific low-level applications or through the workflows API for
high-level, standard applications.
"""

from collections.abc import Iterable
from itertools import chain, product
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils
from scipy.sparse import issparse
from sklearn.preprocessing import OrdinalEncoder


class MetadataConverter:
    """Class used to convert numerical torch tensors into categorical equivalents.

    Allows mapping of numerical tensors with information that would usually be expected
    in :attr:`~anndata.AnnData.obs` into categorical equivalent with literals from the original metadata.
    This class allows for arbitrary subsets of data to be mapped back.

    Parameters
    ----------
    metadata : :class:`~pandas.DataFrame`
        The dataframe object that includes the metadata, which is :attr:`~anndata.AnnData.obs` for most cases.

    Attributes
    ----------
    df_view : :class:`~pandas.DataFrame`
        Dataframe object for reference.

    num_cols : :class:`int`
        Number of columns for `df_view`.
    """

    def __init__(self, metadata_df: pd.DataFrame):
        self.df_view = metadata_df
        self.num_cols = metadata_df.shape[1]

    def _tensor_to_cat(self, met_val_string: torch.Tensor):
        stack_list = []

        for i, colname in enumerate(self.df_view):
            # Decide on single or multi value
            if len(met_val_string.shape) == 1:
                cur_col = met_val_string[i]

            else:
                cur_col = met_val_string[:, i]

            # Do reverse mapping - also decide again on single multi val
            if (
                type(self.df_view[colname].dtype)
                is pd.core.dtypes.dtypes.CategoricalDtype
            ):
                if len(met_val_string.shape) == 1:
                    stack_list.append(
                        self.df_view.iloc[:, i].cat.categories[int(cur_col)]
                    )

                else:
                    stack_list.append(
                        np.array(
                            [
                                self.df_view.iloc[:, i].cat.categories[int(item)]
                                for item in cur_col
                            ]
                        ).reshape(-1, 1)
                    )

            else:
                if len(met_val_string.shape) == 1:
                    stack_list.append(cur_col.numpy())

                else:
                    stack_list.append(cur_col.numpy().reshape(-1, 1))

            i += 1

        return np.hstack(stack_list)

    def map_to_df(self, met_val_string: torch.Tensor) -> np.ndarray:
        """The function that actually does the metadata mapping.

        Parameters
        ----------
        met_val_string : :class:`~torch.Tensor`
            Tensor shaped like :class:`~torch.utils.data.TensorDataset`.

        Returns
        -------
        metadata_mapping : :class:`~numpy.ndarray`
            Numerical tensor converted back to categorical equivalent.
        """
        assert (
            (len(met_val_string.shape) == 2)
            and (met_val_string.shape[1] == self.num_cols)
        ) or (
            (len(met_val_string.shape) == 1)
            and (met_val_string.shape[0] == self.num_cols)
        ), "Input doesn't match defined columns in metadata"

        metadata_mapping = self._tensor_to_cat(met_val_string)
        return metadata_mapping


class AnndataConverter(MetadataConverter):
    """Class used to convert tensor datasets into an :class:`~anndata.AnnData` object.

    This class allows for arbitrary subsets of tensors to be mapped back into
    an object that looks like the subset of the original object. Inherits
    :class:`MetadataConverter`.

    Parameters
    ----------
    metadata_df : :class:`~pandas.DataFrame`
        The dataframe object that includes the metadata, which is :attr:`~anndata.AnnData.obs` for most cases.
    """

    def __init__(self, metadata_df: pd.DataFrame):
        MetadataConverter.__init__(self, metadata_df)

    def map_to_anndat(self, val_tup) -> ad.AnnData:
        """Function to actually do the mapping to :class:`~anndata.AnnData`.

        Parameters
        ----------
        val_tup : :class:`tuple`
            Size 3 tuple of :class:`~torch.Tensor`. The first index is used for counts. The second index
            provides labels, which are not used here as they are redundant but required for training.
            The third index provides the numerically encoded metadata.

        Returns
        -------
        anndat : :class:`~anndata.AnnData`
            The anndata equivalent of the numerical :class:`~torch.Tensor` objects.
        """
        # Make object from the counts
        anndat = ad.AnnData(val_tup[0].numpy())

        # Append metadata to obs, no need for redundant factors in higher level
        df = pd.DataFrame(self.map_to_df(val_tup[2]))
        df.columns = self.df_view.columns

        # Explicit categorical typecasting to play well with metrics
        for colname in df:
            if any(isinstance(value, int | float) for value in df[colname]):
                df[colname] = df[colname].astype(float)

            else:
                df[colname] = df[colname].astype("category")

        anndat.obs = df

        return anndat


class ConcatTensorDataset(utils.ConcatDataset):
    """Allows for arbitrary concatenation of :class:`~torch.utils.data.TensorDataset`.

    Courtesy of https://github.com/johann-petrak/pytorch/commit/eb70e81e31508c383bdc17059ddb532a6b40468c
    ConcatDataset of TensorDatasets which supports getting slices and index lists/arrays.
    This dataset allows the use of slices, e.g. ds[2:4] and of arrays or lists of multiple indices
    if all concatenated datasets are either TensorDatasets or Subset or other ConcatTensorDataset instances
    which eventually contain only TensorDataset instances. If no slicing is needed,
    this class works exactly like torch.utils.data.ConcatDataset and can concatenate arbitrary
    (not just TensorDataset) datasets.

    Parameters
    ----------
        datasets : :class:`Iterable[torch.utils.data.Dataset]`
            List of datasets to be concatenated.
    """

    def __init__(self, datasets: Iterable[utils.Dataset]) -> None:
        super().__init__(datasets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rows = [super().__getitem__(i) for i in range(self.__len__())[idx]]
            return tuple(map(torch.stack, zip(*rows, strict=False)))
        elif isinstance(idx, list | np.ndarray):
            rows = [super().__getitem__(i) for i in idx]
            return tuple(map(torch.stack, zip(*rows, strict=False)))
        else:
            return super().__getitem__(idx)


####################################################################################
############################ Internal Calls ############################
####################################################################################


# Batch processing for distrib_dataset
def _process_batch_dd(dataset):
    l_mean, l_scale = [], []

    for batch in range(int(dataset[:][0][..., -1].view(-1, 1).max().item()) + 1):
        idxs = np.nonzero(dataset[:][0][..., -1] == batch).flatten()
        subset = dataset[list(idxs)][0]
        l_mean.append(subset.sum(-1).log().mean().item())
        l_scale.append(subset.sum(-1).log().var().item())

    return np.array(l_mean), np.array(l_scale)


# Batch processing for construct_labels
def _process_batch_cb(metadata, batch_key):
    encoder = OrdinalEncoder()
    labels = encoder.fit_transform(metadata[batch_key].to_numpy().reshape(-1, 1))
    return encoder, labels


# Get subset indices from cloud
def _get_idxs(point_dataset, target):
    return [
        idx
        for idx in range(len(point_dataset))
        if (point_dataset[idx][1] == target).all()
    ]


# Get the actual subset object from cloud
def _get_subset(point_dataset, target):
    tup = point_dataset[_get_idxs(point_dataset, target)]
    return utils.TensorDataset(*tup)


# Helper to convert metadata from DataFrame to torch object
def _concat_cat_df(metadata):
    stack_list = []

    for colname in metadata:
        if type(metadata[colname].dtype) is pd.core.dtypes.dtypes.CategoricalDtype:
            stack_list.append(metadata[colname].cat.codes.to_numpy().reshape(-1, 1))

        else:
            stack_list.append(metadata[colname].to_numpy().reshape(-1, 1))

    return torch.from_numpy(np.hstack(stack_list)).double()


# Construct combinations of attributes from condition classes
def _factors_to_col(anndat: ad.AnnData, factors: list):
    anndat.obs["factors"] = anndat.obs.apply(
        lambda x: "_".join([x[factor] for factor in factors]), axis=1
    ).astype("category")
    return anndat


# Check to make sure array is dense
def _process_array(arr):
    if isinstance(arr, np.ndarray):  # Check if array is dense
        result = arr

    elif issparse(arr):  # Check if array is sparse
        result = arr.todense()

    else:  # Convert to dense array if not already
        result = np.asarray(arr)

    return result


####################################################################################
############################ Functions ############################
####################################################################################


# Simple preprocessing to conver to anndata
def preprocess_anndata(anndat) -> ad.AnnData:
    """Function to preprocess the inputted :class:`~anndata.AnnData` object.

    Does not do any critical modifications, but instead ensures that the numerical
    columns in :attr:`~anndata.AnnData.obs` are of type :class:`float` and string columns are defined
    as :class:`~pandas.CategoricalDtype`.

    Parameters
    ----------
    anndat : :class:`~anndata.AnnData`
        The :class:`~anndata.AnnData` object to be preprocessed.

    Returns
    -------
    anndat : :class:`~anndata.AnnData`
        Object after typecasting operations.
    """
    for colname in anndat.obs:
        if any(isinstance(value, int | float) for value in anndat.obs[colname]):
            anndat.obs[colname] = anndat.obs[colname].astype(float)

        else:
            anndat.obs[colname] = anndat.obs[colname].astype("category")

    return anndat


def construct_labels(
    counts,
    metadata,
    factors,
    style: Literal["concat", "one-hot"] = "concat",
    batch_key: str = None,
) -> tuple:
    """Function to generate conditional labels for the various models included.

    Parameters
    ----------
    counts
        The field corresponding to :attr:`~anndata.AnnData.X`.

    metadata
        The field corresponding to :attr:`~anndata.AnnData.obs`.

    factors : array_like
        1D Array-like of :class:`str`. The list specifying factors, which are names of the columns from :attr:`~anndata.AnnData.obs`.

    style : :class:`Literal["concat", "one-hot"]`
        Specifies the label encoding.

    batch_key : :class:`str`, optional
        Specifies the batch key, must be included in :attr:`~anndata.AnnData.obs`.

    Returns
    -------
    dataset : :class:`~torch.utils.data.TensorDataset`
        The dataset object to be used downstream.

    levels : :class:`dict`
        A mapping between the literal combinations of `factors` and their numerical equivalents.

    converter : :class:`AnndataConverter`
        The converter object with the associated dataset.

    batch_mapping : :class:`dict`
        Returned only if `batch_key` is given. Ordinal encoding for the batch dimension.
    """
    # Small checks for batch and sparsity
    assert batch_key not in factors, "Batch should not be specified as factor"

    counts = _process_array(counts)

    # Decide on style of labeling:
    # Concat means one-hot attributes will be concatenated
    # One hot means every attribute combination will be considered a single one-hot label

    match style:
        case "concat":
            factors_list = [
                torch.from_numpy(
                    pd.get_dummies(metadata[factor]).to_numpy().astype(int)
                ).double()
                for factor in factors
            ]
            levels = [
                [
                    factor + "_" + elem
                    for elem in list(pd.get_dummies(metadata[factor]).columns)
                ]
                for factor in factors
            ]
            levels_dict = [
                {
                    level[i]: tuple([0] * i + [1] + [0] * (len(level) - 1 - i))
                    for i in range(len(level))
                }
                for level in levels
            ]

            levels_dict_flat = {}
            for d in levels_dict:
                levels_dict_flat.update(d)

            levels_cat = {
                " - ".join(prod): tuple(
                    chain(*[levels_dict_flat[prod[i]] for i in range(len(prod))])
                )
                for prod in product(*[list(level.keys()) for level in levels_dict])
            }

            y = torch.cat(factors_list, dim=-1)

        case "one-hot":
            factors_list = torch.from_numpy(
                pd.get_dummies(metadata.apply(lambda x: " - ".join(x[factors]), axis=1))
                .to_numpy()
                .astype(int)
            ).double()
            cols = list(
                pd.get_dummies(
                    metadata.apply(lambda x: " - ".join(x[factors]), axis=1)
                ).columns
            )
            levels_cat = {
                cols[i]: tuple([0] * i + [1] + [0] * (len(cols) - 1 - i))
                for i in range(len(cols))
            }

            y = factors_list

    # Decide if batch will be appended to input (ie. if working on data that needs batch correction)
    if batch_key is not None:
        encoder, labels = _process_batch_cb(metadata, batch_key)
        x = torch.cat(
            [
                torch.from_numpy(counts),
                torch.from_numpy(labels.astype(int)).double().view(-1, 1),
            ],
            dim=-1,
        )
        return (
            utils.TensorDataset(x, y, _concat_cat_df(metadata)),
            levels_cat,
            AnndataConverter(metadata),
            {
                encoder.categories_[0][t]: t
                for t in range(encoder.categories_[0].shape[0])
            },
        )

    else:
        x = torch.from_numpy(counts).double()
        return (
            utils.TensorDataset(x, y, _concat_cat_df(metadata)),
            levels_cat,
            AnndataConverter(metadata),
        )


# Helper to go from dataset to train-test split loaders
def distrib_dataset(
    dataset: utils.TensorDataset,
    levels: dict,
    split_pcts=None,
    batch_size=128,
    keep_train=None,
    keep_test=None,
    batch_key: str = None,
    **kwargs,
) -> tuple:
    """Function that distributes the :class:`~torch.utils.data.TensorDataset` generated by `construct_labels`.

    Parameters
    ----------
    dataset : :class:`~torch.utils.data.TensorDataset`
        The `dataset` output from `construct_labels`.

    levels : :class:`dict`
        The `levels` output from `construct_labels`.

    split_pcts : array_like, optional
        Size 2 list of `float` specifying the proportions for training and test respectively. Ignored if both `keep_train` and `keep_test` are not `None`.

    batch_size : :class:`int`
        Mini-batch size for the models to train on.

    keep_train : array_like
        1D Array-like of `str`. Specifies the levels to keep in the training dataset. Elements must be from `levels.keys()`.

    keep_test : array_like
        1D Array-like of `str`. Specifies the levels to keep in the test dataset. Elements must be from `levels.keys()`.

    batch_key : :class:`str`, optional
        Must not be `None` if `batch_key` was previously provided to `construct_labels`. The actual values is unimportant for this scope.

    **kwargs : :class:`dict`, optional
        Keyword arguments passed to `utils.DataLoader`.

    Returns
    -------
    train_set : :class:`~torch.utils.data.TensorDataset` or ConcatTensorDataset
        The full training set to be used downstream.

    test_set : :class:`~torch.utils.data.TensorDataset` or ConcatTensorDataset
        The full test set to be used downstream.

    train_loader : :class:`~torch.utils.data.DataLoader`
        The corresponding loader for `train_set`.

    test_loader : :class:`~torch.utils.data.DataLoader`
         The corresponding loader for `test_set`.

    l_mean : :class:`float` or array_like
        If `batch_key` is provided, the empirical library size log-mean for each batch (1-D Array-like of :class:`float`). A single value otherwise.

    l_scale : :class:`float` or array_like
        If `batch_key` is provided, then the empirical library size log-variance for each batch (1-D Array-like of :class:`float`). A single value otherwise.
    """
    if split_pcts is None:
        split_pcts = [0.8, 0.2]

    inv_levels = {v: k for k, v in levels.items()}  # Inverse levels required

    # General training to see how the model fits. USed to evaluate reconstruction or to fit interpretable model with linear decoder.
    if keep_train is None or keep_test is None:
        train_set, test_set = utils.random_split(dataset, split_pcts)
        train_loader, test_loader = (
            utils.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs),
            utils.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs),
        )

    # Used for transfer of conditions. Train test split is completely manually defined and based on attributes
    else:
        print(f"Train Levels: {keep_train}  // Test Levels: {keep_test}")
        train_set = ConcatTensorDataset(
            [
                _get_subset(dataset, torch.tensor(key))
                for key in inv_levels.keys()
                if inv_levels[key] in keep_train
            ]
        )
        test_set = ConcatTensorDataset(
            [
                _get_subset(dataset, torch.tensor(key))
                for key in inv_levels.keys()
                if inv_levels[key] in keep_test
            ]
        )

        train_loader, test_loader = (
            utils.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs),
            utils.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs),
        )

    # If batch is appended to input, generate size priors per batch
    if batch_key is not None:
        l_mean, l_scale = _process_batch_dd(train_set)

    # If not, need a single size prior
    else:
        l_mean, l_scale = (
            train_set[:][0].sum(-1).log().mean(),
            train_set[:][0].sum(-1).log().var(),
        )

    return train_set, test_set, train_loader, test_loader, l_mean, l_scale
