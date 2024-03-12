####################################
###### Assess Point Clouds ##########
####################################

import numpy as np
import numpy as np
import torch
import torch.utils.data as utils
from tqdm import trange
import math



def _norm_lib_size(x, norm_size=1e3):
    lib_sizes = torch.sum(x, dim=1).add(1)
    return torch.div(x.T, lib_sizes).T * norm_size


def _get_normalized_profile(point_set, lib_size=1e3):
    return _norm_lib_size(point_set, lib_size).T.mean(-1)


def get_normalized_profile(point_dataset, target=None, lib_size=1e3):
    
    if target is None:
        point_set = point_dataset[:][0]
    
    else:
        point_set = _get_subset(point_dataset, target)[:][0]
        
    
    return _norm_lib_size(point_set, lib_size).T.mean(-1)

def _get_idxs(point_dataset, target):
    return [idx for idx in range(len(point_dataset)) if (point_dataset[idx][1] == target).all()]

def _get_subset(point_dataset, target):
    tup = point_dataset[_get_idxs(point_dataset, target)]
    return utils.TensorDataset(tup[0], tup[1])

def self_profile_reproduction_error(point_dataset, target=None, n_trials=3000, subset_size=0.5, lib_size=1e3):

    if target is None:
        point_set = point_dataset[:][0]

    else:
        point_set = _get_subset(point_dataset, target)[:][0]
    
    samp_size = int(np.round(point_set.shape[0] * subset_size))
    p_simplex = torch.ones(point_set.shape[0]).div(point_set.shape[0])

    samples = torch.stack([point_set[p_simplex.multinomial(samp_size).tolist()] for i in range(n_trials)])
    profiles = torch.stack([_get_normalized_profile(s, lib_size) for s in samples])

    return profiles, samples


def gen_profile_reproduction_error(point_dataset, model, source, target, n_trials=3000, lib_size=1e3, verbose=False):
    source_set, target_set = _get_subset(point_dataset, source), _get_subset(point_dataset, target)

    if verbose:
        it = trange(n_trials)

    else:
        it = range(n_trials)
    
    preds = torch.stack([model(source_set[:][0].cuda(), target_set[0][1].repeat(len(source_set),1).cuda())['x'][0].cpu() for i in it])
    profiles = torch.stack([_get_normalized_profile(pred, lib_size) for pred in preds])

    return profiles, preds


def get_weighted_reproduction_error(point_dataset, model, source, target, n_trials=3000, subset_size=0.5, lib_size=1e3, verbose=False):
    mean_profile = get_normalized_profile(point_dataset, target=target)

    repr_profiles, samples = self_profile_reproduction_error(point_dataset, target=target, subset_size=subset_size, n_trials=n_trials, lib_size=lib_size)
    repr_mean_rmse = repr_profiles.add(-1*mean_profile).square().mean(-1).mean().item()

    pred_profiles, preds = gen_profile_reproduction_error(point_dataset, model, source, target, n_trials=n_trials, verbose=verbose, lib_size=lib_size)
    preds_mean_rmse = pred_profiles.add(-1*mean_profile).square().mean(-1).mean().item()

    weighted_repr_error = preds_mean_rmse / repr_mean_rmse

    return weighted_repr_error


