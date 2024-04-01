####################################
### Tools to use with real data ########
####################################

import torch
import numpy as np
import pandas as pd
import torch.utils.data as utils
from itertools import combinations, product, permutations, chain
from typing import Iterable, Literal


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



def _get_idxs(point_dataset, target):
    return [idx for idx in range(len(point_dataset)) if (point_dataset[idx][1] == target).all()]


def _get_subset(point_dataset, target):
    tup = point_dataset[_get_idxs(point_dataset, target)]
    return utils.TensorDataset(tup[0], tup[1])


def construct_labels(counts, metadata, factors, style : Literal["concat", "one-hot"] = "concat"):

    match style:
        case "concat":
            
            factors_list = [torch.from_numpy(pd.get_dummies(counts.index.map(lambda x :metadata.loc[x][factor])).to_numpy().astype(int)).double() for factor in factors]
            levels = [[factor + "_" + elem for elem in list(pd.get_dummies(counts.index.map(lambda x :metadata.loc[x][factor])).columns)] for factor in factors]
            levels_dict = [{level[i] : tuple([0]*i + [1] + [0]*(len(level)-1-i)) for i in range(len(level)) } for level in levels]

            levels_dict_flat = {}
            for d in levels_dict:
                levels_dict_flat.update(d)


            levels_cat = {" - ".join(prod) : tuple(chain(*[levels_dict_flat[prod[i]] for i in range(len(prod))])) for prod in product(*[list(level.keys()) for level in levels_dict])}

    

            y = torch.cat(factors_list, dim = -1)
            x = torch.from_numpy(counts.to_numpy()).double()

        case "one-hot":
            
            factors_list = torch.from_numpy(pd.get_dummies(metadata.apply(lambda x : " - ".join(x[factors]), axis=1)).to_numpy().astype(int)).double()
            cols = list(pd.get_dummies(metadata.apply(lambda x : " - ".join(x[factors]), axis=1)).columns)
            levels_cat = { cols[i] : tuple([0]*i + [1] + [0]*(len(cols)-1-i)) for i in range(len(cols))}

            y = factors_list
            x = torch.from_numpy(counts.to_numpy()).double()
            

    return utils.TensorDataset(x,y), levels_cat


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

            
            
          