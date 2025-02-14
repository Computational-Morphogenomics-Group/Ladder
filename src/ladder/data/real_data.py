"""The real_data module houses the functions to be used with real datasets.

The functions defined here can either be used independently for
specific low-level applications or through the workflows API for
high-level, standard applications.
"""

import warnings
from collections.abc import Callable
from functools import partial
from itertools import chain, product
from typing import Literal

import anndata as ad
import numpy as np
import pandas
import pandas as pd
import torch.utils.data as utils
from scipy.sparse import issparse
from sklearn.preprocessing import OrdinalEncoder


class MetadataConverter:
    """Class used to convert numerical torch tensors into categorical equivalents.

    Allows mapping of numerical arrays with information that would usually be expected
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
        self.df_view = self._preprocess_df(metadata_df)
        self.arr_view = self._df_to_arr(self.df_view)
        self.num_cols = metadata_df.shape[1]

    @staticmethod
    def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        for colname in df:
            if any(isinstance(value, int | float) for value in df[colname]):
                df[colname] = df[colname].astype(np.float64)

            else:
                df[colname] = df[colname].astype("category")

        return df

    def _match_coltypes(self, metadata_df: pd.DataFrame):
        for colname in metadata_df.columns:
            metadata_df[colname] = metadata_df[colname].astype(
                self.df_view[colname].dtype
            )

        return metadata_df

    def _df_to_arr(self, metadata_df: pd.DataFrame) -> np.ndarray:
        stack_list = []

        for colname in metadata_df:
            if (
                type(metadata_df[colname].dtype)
                is pd.core.dtypes.dtypes.CategoricalDtype
            ):
                stack_list.append(
                    metadata_df[colname].cat.codes.to_numpy().reshape(-1, 1)
                )

            else:
                stack_list.append(metadata_df[colname].to_numpy().reshape(-1, 1))

        return np.hstack(stack_list).astype(np.float64)

    def _arr_to_df(self, met_val_string: np.ndarray) -> pandas.DataFrame:
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
                    stack_list.append(cur_col)

                else:
                    stack_list.append(cur_col.reshape(-1, 1))

        df = pd.DataFrame(np.hstack(stack_list))

        if len(met_val_string.shape) == 1:
            df = df.T

        df.columns = self.df_view.columns
        return self._match_coltypes(df)

    def map_to_df(self, met_val_string: np.ndarray = None) -> pandas.DataFrame:
        """Mapping function from numeric array to string metadata. Defaults to :attr:`~MetadataConverter.arr_view`

        Parameters
        ----------
        met_val_string : :class:`~numpy.ndarray`, optional
            Tensor shaped like :class:`~ConditionalDataset`.

        Returns
        -------
        metadata_mapping : :class:`~pandas.DataFrame`
            Numerical tensor converted to its categorical equivalent.
        """
        if met_val_string is None:
            met_val_string = self.arr_view

        assert (
            (len(met_val_string.shape) == 2)
            and (met_val_string.shape[1] == self.num_cols)
        ) or (
            (len(met_val_string.shape) == 1)
            and (met_val_string.shape[0] == self.num_cols)
        ), "Input doesn't match defined columns in metadata"

        return self._arr_to_df(met_val_string)

    def map_to_arr(self, metadata_df: pandas.DataFrame = None) -> np.ndarray:
        """Mapping function from string metadata to numeric array. Defaults to :attr:`~MetadataConverter.df_view`

        Parameters
        ----------
        metadata_df : :class:`~pandas.DataFrame`, optional
             :class:`~pandas.DataFrame` matching in content with the metadata used for the constructor.

        Returns
        -------
        numeric_mapping : :class:`~numpy.ndarray`
            Categorical tensor converted to its numerical equivalent.
        """
        if metadata_df is None:
            metadata_df = self.df_view

        assert (metadata_df.columns == self.df_view.columns).all()

        return self._df_to_arr(metadata_df)


class AnndataConverter(MetadataConverter):
    """Class used to convert array datasets into an :class:`~anndata.AnnData` object.

    This class allows for arbitrary subsets of arrays to be mapped back into
    an object that looks like the subset of the original object. Inherits
    :class:`MetadataConverter`.

    Parameters
    ----------
    metadata_df : :class:`~pandas.DataFrame`
        The dataframe object that includes the metadata, which is :attr:`~anndata.AnnData.obs` for most cases.
    """

    def __init__(self, metadata_df: pd.DataFrame):
        MetadataConverter.__init__(self, metadata_df)

    def map_to_anndata(self, val_tup: tuple) -> ad.AnnData:
        """Function to map array view to :class:`~anndata.AnnData`.

        Parameters
        ----------
        val_tup : :class:`tuple`
            Size 3 tuple of array-like. The first index is used for counts. The second index
            provides labels, which are not used here as they are redundant but required for training.
            The third index provides the numerically encoded metadata.

        Returns
        -------
        anndat : :class:`~anndata.AnnData`
            The anndata equivalent of the numerical :class:`~np.ndarray` objects.
        """
        # Make object from the counts
        anndata = ad.AnnData(np.array(val_tup[0]))

        # Append metadata to obs, no need for redundant factors in higher level
        anndata.obs = self.map_to_df(np.array(val_tup[2]))

        return anndata


class ConditionalDataset(utils.Dataset):
    """An override of :class:`~torch.utils.data.Dataset` that offers control over which data type to output.

    Parameters
    ----------
    counts :  :class:`~numpy.ndarray`
        Gene count matrix, assumed to be specifically of type  :class:`~numpy.ndarray`.

    labels :  :class:`~numpy.ndarray`
        Encoded conditional labels, assumed to be specifically of type  :class:`~numpy.ndarray`.

    counts_transform : :class:`~typing.Callable`
        Function to apply to gene counts before passing to model.

    labels_transform : :class:`~typing.Callable`
        Function to apply to conditional labels before passing to model.

    Notes
    -----
    Transforms can be used to change dtype passed to models, making this dataset framework agnostic.

    """

    def __init__(
        self,
        counts,
        labels,
        metadata,
        converter,
        counts_transform: Callable = np.array,
        labels_transform: Callable = np.array,
    ):
        assert len(counts) == len(labels)
        self.counts = counts
        self.labels = labels
        self.metadata = metadata
        self.converter = converter

        self.counts_transform = counts_transform
        self.labels_transform = labels_transform

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        count = self.counts[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]

        count = self.counts_transform(count)
        label = self.labels_transform(label)

        return count, label, metadata

    def __add__(self, other):
        # minor TODO : Add converter equality somehow if needed
        counts = np.vstack((self.counts, other.counts))
        labels = np.vstack((self.labels, other.labels))
        metadata = np.vstack((self.metadata, other.metadata))

        if (
            self.counts_transform != other.counts_transform
            or self.labels_transform != other.labels_transform
        ):
            warnings.warn(f"Transform mismatch detected in {self}", stacklevel=2)

        return ConditionalDataset(
            counts,
            labels,
            metadata,
            self.converter,
            self.counts_transform,
            self.labels_transform,
        )


####################################################################################
############################ Functions ############################
####################################################################################
def construct_labels(
    counts,
    metadata,
    factors,
    style: Literal["concat", "one-hot"] = "concat",
    batch_key: str = None,
    counts_transform: Callable = np.array,
    labels_transform: Callable = np.array,
) -> tuple:
    """Function to generate conditional labels for the various models included.

    Parameters
    ----------
    counts : array-like
        The field corresponding to :attr:`~anndata.AnnData.X`.

    metadata : array-like
        The field corresponding to :attr:`~anndata.AnnData.obs`.

    factors : array_like
        1D Array-like of :class:`str`. The list specifying factors, which are names of the columns from :attr:`~anndata.AnnData.obs`.

    style : :class:`~typing.Literal["concat", "one-hot"]`
        Specifies the label encoding.

    batch_key : :class:`str`, optional
        Specifies the batch key, must be included in :attr:`~anndata.AnnData.obs`.

    Returns
    -------
    dataset : :class:`ConditionalDataset`
        The dataset object to be used downstream.

    levels : :class:`dict`
        A mapping between the literal combinations of `factors` and their numerical equivalents.

    converter : :class:`AnndataConverter`
        The converter object with the associated dataset.

    batch_mapping : :class:`dict`
        Returned only if `batch_key` is given. Ordinal encoding for the batch dimension.
    """

    # Check to make sure array is dense
    def _process_array(arr):
        if isinstance(arr, np.ndarray):  # Check if array is dense
            result = arr

        elif issparse(arr):  # Check if array is sparse
            result = arr.todense()

        else:  # Convert to dense array if not already
            result = np.asarray(arr)

        return np.array(result)

    # Small checks for batch and sparsity
    assert batch_key not in factors, "Batch should not be specified as factor"

    counts = _process_array(counts)

    # Decide on style of labeling:
    # Concat means one-hot attributes will be concatenated
    # One hot means every attribute combination will be considered a single one-hot label

    match style:
        case "concat":
            factors_list = [
                pd.get_dummies(metadata[factor]).to_numpy().astype(np.float64)
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

            y = np.concatenate(factors_list, axis=-1)

        case "one-hot":
            levels = [
                "_".join(elem)
                for elem in product(
                    *list(metadata[factors].apply(lambda x: set(x.unique())))
                )
            ]

            levels_cat = {
                levels[i]: tuple([0] * i + [1] + [0] * (len(levels) - 1 - i))
                for i in range(len(levels))
            }

            y = np.vstack(
                metadata.apply(
                    lambda x: np.array(levels_cat["_".join(list(x[factors]))]),
                    axis=1,
                )
            ).astype(np.float64)

    # Set converter object
    converter = AnndataConverter(metadata)

    # Decide if batch will be appended to input (ie. if working on data that needs batch correction)
    if batch_key is not None:

        # Batch processing for construct_labels
        def _process_batch_cb(metadata, batch_key):
            encoder = OrdinalEncoder()
            labels = encoder.fit_transform(
                metadata[batch_key].to_numpy().reshape(-1, 1)
            )
            return encoder, labels

        encoder, labels = _process_batch_cb(metadata, batch_key)
        x = np.concatenate(
            [
                counts,
                labels,
            ],
            axis=-1,
        )
        return (
            ConditionalDataset(
                x,
                y,
                converter.map_to_arr(),
                converter,
                counts_transform,
                labels_transform,
            ),
            levels_cat,
            {
                encoder.categories_[0][t]: t
                for t in range(encoder.categories_[0].shape[0])
            },
        )

    else:
        x = counts.astype(np.float64)
        return (
            ConditionalDataset(
                x,
                y,
                converter.map_to_arr(),
                converter,
                counts_transform,
                labels_transform,
            ),
            levels_cat,
        )


# Helper to go from dataset to train-test split loaders
def distrib_dataset(
    dataset: ConditionalDataset,
    levels: dict = None,
    split_pcts=None,
    batch_size=128,
    keep_train=None,
    keep_test=None,
    batch_key: str = None,
    **kwargs,
) -> tuple:
    """Function that distributes the :class:`ConditionalDataset` generated by `construct_labels`.

    Parameters
    ----------
    dataset : :class:`ConditionalDataset`
        The `dataset` output from `construct_labels`.

    levels : :class:`dict`, optional
        The `levels` output from `construct_labels`. Required if one of `keep_train` or `keep_test` is not `None`.

    split_pcts : array_like, optional
        Size 2 list of `float` specifying the proportions for training and test respectively. Ignored if both `keep_train` and `keep_test` are not `None`.

    batch_size : :class:`int`
        Mini-batch size for the models to train on.

    keep_train : array_like, optional
        1D Array-like of `str`. Specifies the levels to keep in the training dataset. Elements must be from `levels.keys()`.

    keep_test : array_like, optional
        1D Array-like of `str`. Specifies the levels to keep in the test dataset. Elements must be from `levels.keys()`.

    batch_key : :class:`str`, optional
        Must not be `None` if `batch_key` was previously provided to `construct_labels`. The actual values is unimportant for this scope.

    **kwargs : :class:`dict`, optional
        Keyword arguments passed to `utils.DataLoader`.

    Returns
    -------
    train_set : :class:`ConditionalDataset`
        The full training set to be used downstream.

    test_set : :class:`ConditionalDataset`
        The full test set to be used downstream.

    train_loader : :class:`~torch.utils.data.DataLoader`
        The corresponding loader for `train_set`.

    test_loader : :class:`~torch.utils.data.DataLoader`
         The corresponding loader for `test_set`.

    l_mean : :class:`float` or :class:`~numpy.ndarray`
        If `batch_key` is provided, the empirical library size log-mean for each batch (1-D Array-like of :class:`float`). A single value otherwise.

    l_scale : :class:`float` or :class:`~numpy.ndarray`
        If `batch_key` is provided, then the empirical library size log-variance for each batch (1-D Array-like of :class:`float`). A single value otherwise.
    """

    def collate_fn(batch, counts_transform=np.array, labels_transform=np.array):
        counts, labels, metadata = [], [], []

        for elem in batch:
            counts.append(elem[0])
            labels.append(elem[1])
            metadata.append(elem[2])

        return (
            counts_transform(np.vstack(counts)),
            labels_transform(np.vstack(labels)),
            np.vstack(metadata),
        )

    collator = partial(
        collate_fn,
        counts_transform=dataset.counts_transform,
        labels_transform=dataset.labels_transform,
    )

    if split_pcts is None:
        split_pcts = [0.8, 0.2]

    # General training to see how the model fits. USed to evaluate reconstruction or to fit interpretable model with linear decoder.
    if keep_train is None or keep_test is None:
        train_set, test_set = utils.random_split(dataset, split_pcts)

        train_set, test_set = ConditionalDataset(
            *[np.array(arr) for arr in train_set[:]],
            converter=dataset.converter,
            counts_transform=dataset.counts_transform,
            labels_transform=dataset.labels_transform,
        ), ConditionalDataset(
            *[np.array(arr) for arr in test_set[:]],
            converter=dataset.converter,
            counts_transform=dataset.counts_transform,
            labels_transform=dataset.labels_transform,
        )

        train_loader, test_loader = (
            utils.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                **kwargs,
                collate_fn=collator,
            ),
            utils.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                **kwargs,
                collate_fn=collator,
            ),
        )

    # Used for transfer of conditions. Train test split is completely manually defined and based on attributes
    else:
        inv_levels = {v: k for k, v in levels.items()}  # Inverse levels required

        # Get the actual subset object from cloud
        def _get_subset(point_dataset, target):

            # Get subset indices from cloud
            def _get_idxs(point_dataset, target):
                return [
                    idx
                    for idx in range(len(point_dataset))
                    if (point_dataset.labels[idx] == target).all()
                ]

            tup = point_dataset[_get_idxs(point_dataset, target)]

            return ConditionalDataset(
                *[np.array(arr) for arr in tup],
                converter=point_dataset.converter,
                counts_transform=point_dataset.counts_transform,
                labels_transform=point_dataset.labels_transform,
            )

        print(f"Train Levels: {keep_train}  // Test Levels: {keep_test}")

        def _sum_datasets(datasets):
            base = datasets[0]

            for i in range(1, len(datasets)):
                base = base + datasets[i]

            return base

        train_set = _sum_datasets(
            [
                _get_subset(dataset, np.array(key))
                for key in inv_levels.keys()
                if inv_levels[key] in keep_train
            ]
        )

        test_set = _sum_datasets(
            [
                _get_subset(dataset, np.array(key))
                for key in inv_levels.keys()
                if inv_levels[key] in keep_test
            ]
        )

        train_loader, test_loader = (
            utils.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                **kwargs,
                collate_fn=collator,
            ),
            utils.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                **kwargs,
                collate_fn=collator,
            ),
        )

    # If batch is appended to input, generate size priors per batch
    if batch_key is not None:

        # Batch processing for distrib_dataset
        def _process_batch_dd(dset):
            l_mean, l_scale = [], []

            for batch in range(int(np.max(dset.counts[..., -1])) + 1):
                idxs = np.nonzero(dset.counts[..., -1] == batch)[0]
                subset = dset.counts[list(idxs)]
                l_mean.append(np.mean(np.log(np.sum(np.array(subset), axis=-1))))
                l_scale.append(np.var(np.log(np.sum(np.array(subset), axis=-1))))

            return l_mean, l_scale

        l_mean, l_scale = _process_batch_dd(train_set)

    # If not, need a single size prior
    else:
        l_mean, l_scale = (
            np.mean(np.log(np.sum(np.array(train_set[:][0]), axis=-1))),
            np.var(np.log(np.sum(np.array(train_set[:][0]), axis=-1))),
        )

    return train_set, test_set, train_loader, test_loader, l_mean, l_scale
