####################################
###### Training Utilities ##########
####################################

import torch, pyro
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from pyro.distributions.util import broadcast_shape
from typing import Literal
from pyro.optim import MultiStepLR
import torch.optim as opt
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm
import numpy as np

# Helper to get device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to train basic linear regression
def train_lin_reg(model, train_loader, test_loader, learning_rate=1e-3, epochs=1000, device=get_device()):

    model = model.double().to(device)
    
    loss = torch.nn.MSELoss() 
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(epochs):
        train_loss, test_loss = [], []


        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            
            out = model(x)
            loss_val = loss(out, y)

            train_loss.append(loss_val.item())

            

            opt.zero_grad()
            loss_val.backward()
            opt.step()


        model.eval()
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)

                out = model(x)
                loss_val = loss(out, y)

                test_loss.append(loss_val.item())


        print(f"Epoch : {epoch + 1} // Train Loss : {np.mean(train_loss)} // Test Loss : {np.mean(test_loss)}")

    return model

                  
# Helper to train Pyro models
def train_pyro(model, train_loader, test_loader, num_epochs=1500, verbose=True, device=get_device(), optim_args = {'optimizer': opt.Adam, 'optim_args': {'lr': 4e-4, 'eps' : 1e-2}, 'gamma': 1, 'milestones': [1e10]}):
    
    model = model.double().to(device)
    scheduler = MultiStepLR(optim_args.copy())
    elbo = Trace_ELBO()
    svi = SVI(model.model, model.guide, scheduler, elbo)



    loss_track_test, loss_track_train = [], []

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

    return model, loss_track_train, loss_track_test




# Helper to train models that involve disjoint parameters during training
def train_pyro_disjoint_param(model, train_loader, test_loader, num_epochs=1500, verbose=True, device=get_device(), lr=1e-3, eps=1e-2, style : Literal["joint", "disjoint"] = "disjoint", warmup=0):

    model = model.double().to(device)
    loss_track_test, loss_track_train = [], []
    
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
    
    
    optimizer_nonc = torch.optim.Adam(params_nonc, lr=lr, eps=eps, betas=(0.90, 0.999))
    optimizer_c  = torch.optim.Adam(params_c, lr=lr, eps=eps, betas=(0.90, 0.999))
    
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

    
    
    return model, loss_track_train, loss_track_test, params_nonc_names, params_c_names