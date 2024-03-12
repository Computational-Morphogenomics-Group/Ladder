####################################
###### Training Utilities ##########
####################################

import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from pyro.distributions.util import broadcast_shape
from typing import Literal
from pyro.optim import MultiStepLR
import torch.optim as opt
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO
from tqdm import tqdm
import numpy as np

# Helper to get device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

                  
# Helper to train Pyro models            
def train_pyro(model, train_loader, test_loader, num_epochs=1500, verbose=True, device=get_device(), optim_args = {'optimizer': opt.Adam, 'optim_args': {'lr': 4e-4, 'eps' : 1e-2}, 'gamma': 1, 'milestones': [1e10]}):
    
    model = model.double().to(device)
    scheduler = MultiStepLR(optim_args.copy())
    guide = config_enumerate(model.guide, "parallel", expand=True)
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(model.model, guide, scheduler, elbo)



    loss_track_test, loss_track_train = [], []

    if verbose:
        num_epochs = range(num_epochs)
    else:
        num_epochs = tqdm(range(num_epochs))

    for epoch in num_epochs:
        losses = []
        losses_test = []

        model.train()
    
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = svi.step(x, y)
            losses.append(loss)

    
        model.eval()
        with torch.no_grad(): 
            for x,y in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss = elbo.loss(model.model, model.guide, x,y)
                losses_test.append(test_loss)
            

    
        scheduler.step()

        if verbose:
            print(f"Epoch : {epoch} || Train Loss: {np.mean(losses).round(5)} || Test Loss: {np.mean(losses_test).round(5)}")

    loss_track_train.append(np.mean(losses))
    loss_track_test.append(np.mean(losses_test))

    return model, loss_track_train, loss_track_test
          