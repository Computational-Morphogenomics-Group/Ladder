####################################
## Basic model components ##########
####################################

import torch
import torch.nn as nn
from pyro.distributions.util import broadcast_shape
from typing import Literal


def save_model(model, file_path):
    """
    Saves the given model to disk.

    Parameters
    ----------
    model : torch.nn.Module
        model object to save
    file_path : str
        path to save model state

        
    Returns
    -------
    None
    
    Examples
    --------
    >>> from wh.models.basics import save_model
    >>> save_model(model, "save_path.pth")
    """
    torch.save(model.state_dict(), file_path)


def load_model(model_class, file_path, *args, **kwargs):
    """
    Loads the given model class with parameters from disk.

    Parameters
    ----------
    model_class : torch.nn.Module
        model class - not the object - to be loaded
    file_path : str
        path to save model state
    args : any
        args to pass to model instantion
    kwargs : any
        kwargs to pass to model instantion

        
    Returns
    -------
    model : torch.nn.Module
        the instantiated object with parameters filled 
    
    Examples
    --------
    >>> from wh.models.basics import load_model
    >>> from wh.models.condvaes import CondVAE
    >>> load_model(CondVAE, "save_path.pth", 3, 2, latent_size=10, betas=[20,0.2])

    CondVAE(...
    """
    
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(file_path))
    return model


class MLP(nn.Module):
    """
    Basic MLP to be used within other models.

    Parameters
    ----------
    input_size : int
        size of the input layer
    hidden_sizes : list[int]
        list of sizes to be used in the fully connected layers
    output_size : int
        size of the output layer
    final_activation : torch.nn.Module
        non-parametric function to be applied to the output 
    gain : float
        higher gain will instantiate model with larger parameters
        

    
    Examples
    --------
    >>> from wh.models.basics import MLP
    >>> MLP(32, [32]*3, 32)
    
    MLP(
      (fc_layers): ModuleList(
        (0): Linear(in_features=32, out_features=32, bias=True)
        (1): ReLU()
        (2): Linear(in_features=32, out_features=32, bias=True)
        (3): ReLU()
        (4): Linear(in_features=32, out_features=32, bias=True)
      )
    )

    """
    def __init__(self, input_size, hidden_sizes=[256,256], output_size=32, final_activation=None, bias=True, gain=0.1):
        super(MLP, self).__init__()

        # To keep all hidden layers
        self.fc_layers = nn.ModuleList()

        # Input to hidden layers
        self.fc_layers.append(nn.Linear(input_size, hidden_sizes[0], bias=bias))
        self.fc_layers.append(nn.ReLU())
        
        # Between hidden layers
        for i in range(1, len(hidden_sizes)):
            self.fc_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=bias))
            self.fc_layers.append(nn.ReLU())

        # Hidden to output
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], output_size, bias=bias))

        # Init layers
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=gain)

        # Might need to add activation for final layer
        
        if isinstance(final_activation, nn.Module):
            self.fc_layers.append(final_activation)

        elif final_activation:
            print("Non-module given for final activation, ignored...")


    
    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x



## Helpers taken from https://pyro.ai/examples/scanvi.html

def split_in_half(t):
    """
    Splits a tensor in half along the final dimension
    """
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


def broadcast_inputs(input_args):
    """
    Helper for broadcasting inputs to neural net
    """
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args


def make_fc(dims):
    """
    Helper to make FC layers in succession
    """
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])


def enc_dec(in_dims, hidden_dims, out_dims, last_config=0, dist_config : Literal["normal", "zinb", "categorical", "normal+lognormal", "classifier"] = "normal"):
    """
    Helper to construct NN functions
    """
    
    def __init__(self, in_dims, hidden_dims, out_dims):
        super().__init__()

        match last_config:

            case 0: # Last will be 2*out for easy reparam
                dims = [in_dims] + hidden_dims + [2 * out_dims]

            case _: # Last will include +2 for l_loc & l_scale
                dims = [in_dims] + hidden_dims + [2*out_dims + 2]
                
                
        self.fc = make_fc(dims)

    def forward(self, inputs):
        
        match dist_config:
            case "zinb": # For decoders
                gate_logits, mu = split_in_half(self.fc(inputs))
                
                gate_logits = torch.clamp(gate_logits, min=1e-5, max=1-1e-5)
                mu = softmax(mu, dim=-1)
                
                return gate_logits, mu

            case "classifier": # For discriminators
                logits = self.fc(inputs)
                return logits

            case "normal": # For count precursors 
                ## Pre-conditions below
                
                # With broadcast
                #z2_y = broadcast_inputs([z2, y])
                #z2_y = torch.cat(z2_y, dim=-1)

                # Without broadcast
                #inputs = torch.cat([z1, y], dim=-1) must be satisfied
            
                _inputs= inputs.reshape(-1, inputs.size(-1))
                hidden = self.fc(_inputs)
                hidden = hidden.reshape(inputs.shape[:-1] + hidden.shape[-1:])
                
                loc, scale = split_in_half(hidden)
                scale = softplus(scale)
                
                return loc, scale

            case "normal+lognormal": # For counts
                inputs = torch.log(1 + inputs)
                h1, h2 = split_in_half(self.fc(inputs))
                
                norm_loc, norm_scale = h1[..., :-1], softplus(h2[..., :-1])
                l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:])
                
                return z2_loc, z2_scale, l_loc, l_scale

      
                  
                    
                    
                


    
        