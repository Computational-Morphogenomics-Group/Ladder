####################################
## Basic model components ##########
####################################

import torch
import torch.nn as nn

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
    def __init__(self, input_size, hidden_sizes=[256,256], output_size=32, final_activation=None, bias=True, gain=1):
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