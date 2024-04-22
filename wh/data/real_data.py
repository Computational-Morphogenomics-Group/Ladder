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




# Helper to convert between numeric and categorical views of metadata
# tensor can be a subset of the original metadata tensor
# metadata should be the full original metadata object at all times to preserve category encoding
class MetadataConverter:

    def __init__(self, metadata_df : pd.DataFrame):
        
        self.df_view = metadata_df
        self.num_cols = metadata_df.shape[1]


    def _tensor_to_cat(self, met_val_string : torch.Tensor):

        stack_list = []

        for i, colname in enumerate(self.df_view):
            # Decide on single or multi value
            if len(met_val_string.shape) == 1:
                cur_col = met_val_string[i]

            else:
                cur_col = met_val_string[:,i]

        
            # Do reverse mapping - also decide again on single multi val     
            if type(self.df_view[colname].dtype) == pd.core.dtypes.dtypes.CategoricalDtype:
                if len(met_val_string.shape) == 1:
                    stack_list.append(self.df_view.iloc[:,i].cat.categories[int(cur_col)])
                
                else:
                    stack_list.append(np.array([self.df_view.iloc[:,i].cat.categories[int(item)] for item in cur_col]).reshape(-1,1))
            
            
            else:
                if len(met_val_string.shape) == 1:
                    stack_list.append(cur_col.numpy())
                
                else:
                    stack_list.append(cur_col.numpy().reshape(-1,1))

            i += 1
    
        return np.hstack(stack_list)
        
            


    def map_to_df(self, met_val_string : torch.Tensor):
        assert ( (len(met_val_string.shape) == 2) and (met_val_string.shape[1] == self.num_cols) ) or ( (len(met_val_string.shape) == 1) and (met_val_string.shape[0] == self.num_cols) ), "Input doesn't match defined columns in metadata"

        return self._tensor_to_cat(met_val_string)




# Helper to convert tuple torch tensors into anndata views for further use
# tensor tuples are expected as (counts, labels, metadata)
# the original metadata df is also needed to preserve categories
class AnndataConverter(MetadataConverter):

    def __init__(self, metadata_df : pd.DataFrame):
         MetadataConverter.__init__(self, metadata_df)

    
    def map_to_anndat(self, val_tup):

        # Make object from the counts
        anndat = ad.AnnData(val_tup[0].numpy())

        # Append metadata to obs, no need for redundant factors in higher level
        df = pd.DataFrame(self.map_to_df(val_tup[2]))
        df.columns = self.df_view.columns
        anndat.obs = df

        return anndat

        
    




class ConcatTensorDataset(utils.ConcatDataset):
    r"""ConcatDataset of TensorDatasets which supports getting slices and index lists/arrays.
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
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in range(self.__len__())[idx]]
            return tuple(map(torch.stack, zip(*rows)))
        elif isinstance(idx, (list, np.ndarray)):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in idx]
            return tuple(map(torch.stack, zip(*rows)))
        else:
            return super(ConcatTensorDataset, self).__getitem__(idx)



############################ Internal Calls ############################
def _get_idxs(point_dataset, target):
    return [idx for idx in range(len(point_dataset)) if (point_dataset[idx][1] == target).all()]


def _get_subset(point_dataset, target):
    tup = point_dataset[_get_idxs(point_dataset, target)]
    return utils.TensorDataset(tup[0], tup[1])


def _concat_cat_df(metadata):

    stack_list = []

    for colname in metadata:
        if type(metadata[colname].dtype) == pd.core.dtypes.dtypes.CategoricalDtype:
            stack_list.append(metadata[colname].cat.codes.to_numpy().reshape(-1,1))

        else:
            stack_list.append(metadata[colname].to_numpy().reshape(-1,1))

    return torch.from_numpy(np.hstack(stack_list)).double()
        
####################################################################################




############################ Funcitons ############################


# Helper to get dataset for CVAE models
def construct_labels(counts, metadata, factors, style : Literal["concat", "one-hot"] = "concat", inc_batch = False):

    assert "batch" not in factors

    # Pre-process metadata to remove object columns:
    for colname in metadata.dtypes[metadata.dtypes == "object"].index:
        metadata[colname] = metadata[colname].astype("category")


    # Decide on style of labeling:
    # Concat means one-hot attributes will be concatenated
    # One hot means every attribute combination will be considered a single one-hot label
    
    match style:
        case "concat":
            
            factors_list = [torch.from_numpy(pd.get_dummies(counts.index.map(lambda x :metadata.loc[x][factor])).to_numpy().astype(int)).double() for factor in factors]
            levels = [[factor + "_" + elem for elem in list(pd.get_dummies(counts.index.map(lambda x :metadata.loc[x][factor])).columns)] for factor in factors]
            levels_dict = [{level[i] : tuple([0]*i + [1] + [0]*(len(level)-1-i)) for i in range(len(level)) } for level in levels]

            levels_dict_flat = {}
            for d in levels_dict:
                levels_dict_flat.update(d)


            levels_cat = {" - ".join(prod) : tuple(chain(*[levels_dict_flat[prod[i]] for i in range(len(prod))])) for prod in product(*[list(level.keys()) for level in levels_dict])}

            if inc_batch:
                x = torch.cat([torch.from_numpy(counts.to_numpy()), torch.from_numpy(metadata["batch"].astype(int).to_numpy()).double().view(-1,1)], dim=-1)
            
            else:
                x = torch.from_numpy(counts.to_numpy()).double()

                
            y = torch.cat(factors_list, dim = -1)

        case "one-hot":
            
            factors_list = torch.from_numpy(pd.get_dummies(metadata.apply(lambda x : " - ".join(x[factors]), axis=1)).to_numpy().astype(int)).double()
            cols = list(pd.get_dummies(metadata.apply(lambda x : " - ".join(x[factors]), axis=1)).columns)
            levels_cat = { cols[i] : tuple([0]*i + [1] + [0]*(len(cols)-1-i)) for i in range(len(cols))}

            if inc_batch:
                x = torch.cat([torch.from_numpy(counts.to_numpy()), torch.from_numpy(metadata["batch"].astype(int).to_numpy()).double().view(-1,1)], dim=-1)
            
            else:
                x = torch.from_numpy(counts.to_numpy()).double()

            y = factors_list
            

    
    return utils.TensorDataset(x,y, _concat_cat_df(metadata)), levels_cat, AnndataConverter(metadata)



# Helper to go from dataset to train-test split loaders
def distrib_dataset(dataset, levels, split_type : Literal["o_o", "o_u", "u_u", "u"] = "o_o", split_pcts = [0.8, 0.2], batch_size=256, source=(1,0,0,0,1), target=(0,1,0,0,1)):

    np.random.seed(42)
    torch.manual_seed(42)

    inv_levels = {v: k for k, v in levels.items()}

    match split_type:

        case "o_o":
            train_set, test_set = utils.random_split(dataset, split_pcts)
            train_loader, test_loader = utils.DataLoader(train_set, num_workers=4, batch_size=batch_size, shuffle=True), utils.DataLoader(test_set, num_workers=4, batch_size=batch_size, shuffle=False)

        case "o_u" | "u_u" | "u":
            source_set, target_set = _get_subset(dataset, torch.tensor(source)), _get_subset(dataset, torch.tensor(target))

            print(f"Source : {inv_levels[source]} // Target : {inv_levels[target]}")

            inv_levels.pop(target) # Only pop the target
            print("Popped target")

            if split_type == "u_u":
                inv_levels.pop(source) # Pop source as well 
                print("Popped source")


            
            rest = ConcatTensorDataset([_get_subset(dataset, torch.tensor(key)) for key in inv_levels.keys()])

            if split_type != "u":
                target_set = ConcatTensorDataset([source_set, target_set])
            
            train_set, test_set = rest, target_set
            train_loader, test_loader = utils.DataLoader(train_set, num_workers=4, batch_size=batch_size, shuffle=True), utils.DataLoader(test_set, num_workers=4, batch_size=batch_size, shuffle=False)

        
    return train_set, test_set, train_loader, test_loader

            


# Helper to train linear regression with optional matchings
def make_lin_reg_data(counts, metadata, batch_size=128, split_factor="species", group_names = ["0", "1"], split_pcts=[0.8, 0.2], matchings : Literal["random", "ot"] = "random"):
    x,y = counts.loc[metadata.groupby(split_factor, observed=True).get_group(group_names[0]).index], counts.loc[metadata.groupby(split_factor, observed=True).get_group(group_names[1]).index]

    match matchings:
        case "random":
            dataset = utils.TensorDataset(torch.from_numpy(x.to_numpy()).double(), torch.from_numpy(y.sample(len(x)).to_numpy()).double())
            train_set, test_set = utils.random_split(dataset, split_pcts)
            train_loader, test_loader = utils.DataLoader(train_set, num_workers=4, batch_size=batch_size, shuffle=True), utils.DataLoader(test_set, num_workers=4, batch_size=batch_size, shuffle=False)

        case "ot":
            pass

    
    return train_set, test_set, train_loader, test_loader

          