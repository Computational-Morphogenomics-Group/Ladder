####################################
### Tools to use with real data ########
####################################

import torch
import numpy as np
import pandas as pd
import torch.utils.data as utils
from itertools import combinations, product, permutations, chain
from typing import Iterable, Literal
import anndata as ad
from scipy.sparse import issparse, csr_matrix
import ot
from sklearn.preprocessing import OrdinalEncoder


# Helper to convert between numeric and categorical views of metadata
# tensor can be a subset of the original metadata tensor
# metadata should be the full original metadata object at all times to preserve category encoding
class MetadataConverter:

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
                == pd.core.dtypes.dtypes.CategoricalDtype
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

    def map_to_df(self, met_val_string: torch.Tensor):
        assert (
            (len(met_val_string.shape) == 2)
            and (met_val_string.shape[1] == self.num_cols)
        ) or (
            (len(met_val_string.shape) == 1)
            and (met_val_string.shape[0] == self.num_cols)
        ), "Input doesn't match defined columns in metadata"

        return self._tensor_to_cat(met_val_string)


# Helper to convert tuple torch tensors into anndata views for further use
# tensor tuples are expected as (counts, labels, metadata)
# the original metadata df is also needed to preserve categories
class AnndataConverter(MetadataConverter):

    def __init__(self, metadata_df: pd.DataFrame):
        MetadataConverter.__init__(self, metadata_df)

    def map_to_anndat(self, val_tup):

        # Make object from the counts
        anndat = ad.AnnData(val_tup[0].numpy())

        # Append metadata to obs, no need for redundant factors in higher level
        df = pd.DataFrame(self.map_to_df(val_tup[2]))
        df.columns = self.df_view.columns

        # Explicit categorical typecasting to play well with metrics
        for colname in df:
            if any(isinstance(value, (int, float)) for value in df[colname]):
                df[colname] = df[colname].astype(float)

            else:
                df[colname] = df[colname].astype("category")

        anndat.obs = df

        return anndat


class ConcatTensorDataset(utils.ConcatDataset):
    r"""

    Courtesy of https://github.com/johann-petrak/pytorch/commit/eb70e81e31508c383bdc17059ddb532a6b40468c

    ConcatDataset of TensorDatasets which supports getting slices and index lists/arrays.
    This dataset allows the use of slices, e.g. ds[2:4] and of arrays or lists of multiple indices
    if all concatenated datasets are either TensorDatasets or Subset or other ConcatTensorDataset instances
    which eventually contain only TensorDataset instances. If no slicing is needed,
    this class works exactly like torch.utils.data.ConcatDataset and can concatenate arbitrary
    (not just TensorDataset) datasets.
    Args:
        datasets (sequence): List of datasets to be concatenated


    """

    def __init__(self, datasets: Iterable[utils.Dataset]) -> None:
        super(ConcatTensorDataset, self).__init__(datasets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rows = [
                super(ConcatTensorDataset, self).__getitem__(i)
                for i in range(self.__len__())[idx]
            ]
            return tuple(map(torch.stack, zip(*rows)))
        elif isinstance(idx, (list, np.ndarray)):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in idx]
            return tuple(map(torch.stack, zip(*rows)))
        else:
            return super(ConcatTensorDataset, self).__getitem__(idx)


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
        if type(metadata[colname].dtype) == pd.core.dtypes.dtypes.CategoricalDtype:
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
def preprocess_anndata(anndat):
    for colname in anndat.obs:
        if any(isinstance(value, (int, float)) for value in anndat.obs[colname]):
            anndat.obs[colname] = anndat.obs[colname].astype(float)

        else:
            anndat.obs[colname] = anndat.obs[colname].astype("category")

    return anndat


# Helper to get dataset for CVAE models


def construct_labels(
    counts,
    metadata,
    factors,
    style: Literal["concat", "one-hot"] = "concat",
    batch_key=None,
):

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
    dataset,
    levels,
    split_pcts=[0.8, 0.2],
    batch_size=128,
    keep_train=None,
    keep_test=None,
    batch_key=None,
    **kwargs,
):

    inv_levels = {v: k for k, v in levels.items()}  # Inverse levels required

    # General training to see how the model fits. USed to evaluate reconstruction or to fit interpretable model with linear decoder.
    if keep_train is None or keep_test is None:
        train_set, test_set = utils.random_split(dataset, split_pcts)
        train_loader, test_loader = utils.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, **kwargs
        ), utils.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

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

        train_loader, test_loader = utils.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, **kwargs
        ), utils.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

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


# Helper to train linear regression with optional matchings
def make_lin_reg_data(
    counts,
    metadata,
    split_factor,
    labels,
    source_groups,
    target_groups,
    batch_size=128,
    split_pcts=[0.8, 0.2],
    matchings: Literal["random", "ot"] = "random",
):

    # Densify
    counts = _process_array(counts)

    sources, targets = [], []

    for lab in metadata[labels].cat.categories:
        lab_locs = np.where(
            metadata.index.isin(metadata[metadata[labels] == lab].index)
        )
        sub_metadata, sub_counts = metadata.iloc[lab_locs], counts[lab_locs]

        locs_source, locs_target = np.where(
            sub_metadata.index.isin(
                [
                    x
                    for xs in [
                        list(
                            sub_metadata.groupby("factors", observed=True)
                            .get_group(group_name)
                            .index
                        )
                        for group_name in source_groups
                    ]
                    for x in xs
                ]
            )
        ), np.where(
            sub_metadata.index.isin(
                [
                    x
                    for xs in [
                        list(
                            sub_metadata.groupby("factors", observed=True)
                            .get_group(group_name)
                            .index
                        )
                        for group_name in target_groups
                    ]
                    for x in xs
                ]
            )
        )

        x, y = sub_counts[locs_source], sub_counts[locs_target]

        match matchings:
            case "random":
                y = torch.from_numpy(
                    y[np.random.choice(y.shape[0], x.shape[0], replace=True)]
                ).double()

            case "ot":
                # Set up basic ot
                a, b = (
                    np.ones((x.shape[0],)) / x.shape[0],
                    np.ones((y.shape[0],)) / y.shape[0],
                )
                M = ot.dist(x, y, metric="correlation")
                Gs = ot.sinkhorn(a, b, M, 1e-1)

                idxs = [row.argmax() for row in Gs]
                y = torch.from_numpy(y[idxs]).double()

        x = torch.from_numpy(x).double()
        sources.append(x)
        targets.append(y)

    x, y = torch.vstack(sources), torch.vstack(targets)

    dataset = utils.TensorDataset(x, y)
    train_set, test_set = utils.random_split(dataset, split_pcts)
    train_loader, test_loader = utils.DataLoader(
        train_set, num_workers=4, batch_size=batch_size, shuffle=True
    ), utils.DataLoader(test_set, num_workers=4, batch_size=batch_size, shuffle=False)

    return train_set, test_set, train_loader, test_loader
