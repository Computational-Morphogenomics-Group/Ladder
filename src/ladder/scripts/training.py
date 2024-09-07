####################################
###### Training Utilities ##########
####################################

import torch, pyro
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from pyro.distributions.util import broadcast_shape
from typing import Literal
from pyro.optim import MultiStepLR
import pyro.optim as opt
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm
import numpy as np

# Helper to get device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to train Pyro models
def train_pyro(model, train_loader, test_loader, num_epochs=1500, convergence_threshold=1e-3, verbose=True, device=get_device(), optim_args = {'optimizer': opt.Adam, 'optim_args': {'lr': 1e-3, 'eps' : 1e-2}, 'gamma': 1, 'milestones': [1e10]}):
    print(f'Using device: {device}\n')

    model = model.double().to(device)
    scheduler = MultiStepLR(optim_args.copy())
    elbo = Trace_ELBO()
    svi = SVI(model.model, model.guide, scheduler, elbo)



    loss_track_test, loss_track_train, losses_min = [], [], [np.inf]
    min_count = 0

    if verbose:
        num_epochs = range(num_epochs)
    else:
        num_epochs = tqdm(range(num_epochs))

    for epoch in num_epochs:
        losses = []
        losses_test = []

        model.train()
    
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            loss = svi.step(x, y)
            losses.append(loss)

    
        model.eval()
        with torch.no_grad(): 
            for x, y, _ in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss = elbo.loss(model.model, model.guide, x,y)
                losses_test.append(test_loss)
            

    
        scheduler.step()

        if verbose:
            print(f"Epoch : {epoch} || Train Loss: {np.mean(losses).round(5)} || Test Loss: {np.mean(losses_test).round(5)}")

        loss_track_train.append(np.mean(losses))
        loss_track_test.append(np.mean(losses_test))
        min_count += 1
        
        if ((np.min(losses_min) - np.mean(losses_test)) > convergence_threshold): 
            losses_min.append(np.mean(losses_test))
            min_count = 0

        if min_count == 15:
            print(f"Convergence detected with last 15 epochs improvement {losses_min[-1] - np.min(loss_track_test[-15:])}, ending training...")
            break

    
    return model, loss_track_train, loss_track_test




# Helper to train models that involve disjoint parameters during training
def train_pyro_disjoint_param(model, train_loader, test_loader, num_epochs=1500, convergence_threshold=1e-3, verbose=True, device=get_device(), lr=1e-3, eps=1e-2, betas=(0.90, 0.999), style : Literal["joint", "disjoint"] = "disjoint", warmup=0):

    print(f'Using device: {device}\n')

    model = model.double().to(device)
    loss_track_test, loss_track_train, losses_min = [], [], [np.inf]
    min_count = 0
    
    # Defining losses
    loss_fn = lambda model, guide, x, y: pyro.infer.Trace_ELBO().differentiable_loss(model, guide, x, y)
    
    # Params & optims
    x,y,_ = next(iter(train_loader))
    with pyro.poutine.trace(param_only=True) as param_capture:
        loss = loss_fn(model.model, model.guide, x.to(device), y.to(device))

    params_nonc = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values() if "classifier_z" not in site["name"])
    params_nonc_names = set(site["name"] for site in param_capture.trace.nodes.values() if "classifier_z" not in site["name"])

    params_c = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values() if "classifier_z" in site["name"])
    params_c_names = set(site["name"] for site in param_capture.trace.nodes.values() if "classifier_z" in site["name"])
    
    
    optimizer_nonc = torch.optim.Adam(params_nonc, lr=lr, eps=eps, betas=betas)
    optimizer_c  = torch.optim.Adam(params_c, lr=lr, eps=eps, betas=betas)
    
    # Train loop

    if verbose:
        num_epochs = range(num_epochs)
    else:
        num_epochs = tqdm(range(num_epochs))



    
    for epoch in num_epochs :
        losses, prob_losses = [], []
        losses_test, prob_losses_test = [], []

        model.train()

        match style:

            case "disjoint":

                # Classifier trains over dataset
                for x, y, _ in train_loader:
                    x, y = x.to(device), y.to(device)
                    log_prob_loss = model.adverserial(x,y).mean()
                    prob_losses.append(log_prob_loss.detach().cpu())

                    optimizer_c.zero_grad()
                    log_prob_loss.backward()
                    optimizer_c.step()
            

                # Other parameters also train
                if epoch+1 > warmup:
                    for x, y, _ in train_loader:
                        x, y = x.to(device), y.to(device)
                        loss = loss_fn(model.model, model.guide, x, y)
                        losses.append(loss.detach().cpu())

                        optimizer_nonc.zero_grad()
                        loss.backward()
                        optimizer_nonc.step()
            
        
            case "joint":

                # Train at the same time 
                for x, y, _ in train_loader:
                    x, y = x.to(device), y.to(device)


                    # Classifier branch
                    log_prob_loss = model.adverserial(x,y).mean()
                    prob_losses.append(log_prob_loss.detach().cpu())

                    optimizer_c.zero_grad()
                    log_prob_loss.backward()
                    optimizer_c.step()


                    # Other params branch
                    if epoch+1 > warmup:
                        loss = loss_fn(model.model, model.guide, x, y)
                        losses.append(loss.detach().cpu())

                        optimizer_nonc.zero_grad()
                        loss.backward()
                        optimizer_nonc.step()

                        #print(f"logloss: {log_prob_loss} // loss: {loss}")

                
                

        # Testing
        model.eval()
        with torch.no_grad(): 
            for x, y, _ in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss = loss_fn(model.model, model.guide, x,y)
                losses_test.append(test_loss.detach().cpu())

                log_prob_loss = model.adverserial(x,y).mean()
                prob_losses_test.append(log_prob_loss.detach().cpu())
            

    
        #scheduler.step()

        if verbose:
            print(f"Epoch : {epoch} || Train Loss: {np.mean(losses).round(5)} // {np.mean(prob_losses).round(5)} || Test Loss: {np.mean(losses_test).round(5)} // {np.mean(prob_losses_test).round(5)} || Warmup : {bool(epoch+1 <= warmup)}")
        
        loss_track_train.append(np.mean(losses))
        loss_track_test.append(np.mean(losses_test))

        min_count += 1
        
        if ((np.min(losses_min) - np.mean(losses_test)) > convergence_threshold): 
            losses_min.append(np.mean(losses_test))
            min_count = 0

        if min_count == 15:
            print(f"Convergence detected with last 15 epochs improvement {losses_min[-1] - np.min(loss_track_test[-15:])}, ending training...")
            break


    
    return model, loss_track_train, loss_track_test, params_nonc_names, params_c_names
