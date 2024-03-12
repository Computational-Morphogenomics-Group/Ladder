####################################
###### Assess Point Clouds ##########
####################################

import numpy as np
import numpy as np
import torch
import torch.utils.data as utils
from tqdm import trange
import math
from typing import Literal

GAMMA = np.round((1.-math.gamma(1+1.e-8))*1.e14 )*1.e-6

####################################
####################################
##### Inner Calls #####
####################################
####################################



def _solve_coupon_collector(num):
    return int(np.round((num * np.log(num)) + (num*GAMMA)))


def _norm_lib_size(x, norm_size=1e3):
    lib_sizes = torch.sum(x, dim=1).add(1)
    return torch.div(x.T, lib_sizes).T * norm_size


def _get_normalized_profile(point_set, lib_size=1e3):
    return _norm_lib_size(point_set, lib_size).T.mean(-1)


def _get_idxs(point_dataset, target):
    return [idx for idx in range(len(point_dataset)) if (point_dataset[idx][1] == target).all()]


def _get_subset(point_dataset, target):
    tup = point_dataset[_get_idxs(point_dataset, target)]
    return utils.TensorDataset(tup[0], tup[1])


def _get_rmse_n_to_1(profiles, mean_profile):
    return profiles.add(-1*mean_profile).square().mean(-1).sqrt().mean().item()


def _get_chamf_n_to_1(samples, orig):
    matches = [torch.cdist(orig, samples[i], p=2).argmin(-1) for i in range(len(samples))]
    chamf = torch.stack([orig.add(-1*(samples[i][matches[i]])).square().mean() for i in range(len(samples))])
    return chamf.mean().item()

####################################
####################################
#### Functions #####
####################################
####################################


def get_normalized_profile(point_dataset, target=None, lib_size=1e3):
    
    if target is None:
        point_set = point_dataset[:][0]
    
    else:
        point_set = _get_subset(point_dataset, target)[:][0]
        
    
    return _norm_lib_size(point_set, lib_size).T.mean(-1)


def self_profile_reproduction(point_dataset, target=None, n_trials=3000, subset_size=0.5, lib_size=1e3):

    if target is None:
        point_set = point_dataset[:][0]

    else:
        point_set = _get_subset(point_dataset, target)[:][0]
    
    samp_size = int(np.round(point_set.shape[0] * subset_size))
    p_simplex = torch.ones(point_set.shape[0]).div(point_set.shape[0])

    samples = torch.stack([point_set[p_simplex.multinomial(samp_size).tolist()] for i in range(n_trials)])
    profiles = torch.stack([_get_normalized_profile(s, lib_size) for s in samples])

    return profiles, samples



def gen_profile_reproduction(point_dataset, model, source, target, n_trials=3000, lib_size=1e3, verbose=False):
    source_set, target_set = _get_subset(point_dataset, source), _get_subset(point_dataset, target)

    if verbose:
        it = trange(n_trials)

    else:
        it = range(n_trials)
    
    preds = torch.stack([model(source_set[:][0].cuda(), target_set[0][1].repeat(len(source_set),1).cuda())['x'][0].cpu() for i in it])
    profiles = torch.stack([_get_normalized_profile(pred, lib_size) for pred in preds])

    return profiles, preds



def get_weighted_reproduction_error(point_dataset, model, source, target, metric : Literal["chamfer", "rmse"] = "rmse", n_trials=None, subset_size=0.5, lib_size=1e3, verbose=False):

    if n_trials is None:
        print("Defaulting to coupon collector for n_trials...")
        n_trials = _solve_coupon_collector(len(point_dataset))

    match metric: # Add different case for each key
        case "rmse":
            _metric_func = _get_rmse_n_to_1

        case "chamfer":
            _metric_func = _get_chamf_n_to_1
        
    
    repr_profiles, samples = self_profile_reproduction(point_dataset, target=target, subset_size=subset_size, n_trials=n_trials, lib_size=lib_size)
    pred_profiles, preds = gen_profile_reproduction(point_dataset, model, source, target, n_trials=n_trials, verbose=verbose, lib_size=lib_size)

    
    match metric:
        case "rmse": # Add profile metrics here 
            mean_profile = get_normalized_profile(point_dataset, target=target)
            repr_mean_error = _metric_func(repr_profiles, mean_profile)
            preds_mean_error = _metric_func(pred_profiles, mean_profile)
        
        case "chamfer": # Add cloud metrics here
            orig = _get_subset(point_dataset, target)[:][0]
            repr_mean_error = _metric_func(samples, orig)
            preds_mean_error = _metric_func(preds, orig)
            

    
    weighted_repr_error = preds_mean_error / repr_mean_error

    
    return weighted_repr_error, repr_profiles, pred_profiles, samples, preds


