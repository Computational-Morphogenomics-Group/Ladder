####################################
## Basic model components ##########
####################################

import torch
import torch.nn as nn

def save_model(model, file_path):
    """
    Saves the given model to disk.
    """
    torch.save(model.state_dict(), file_path)


def load_model(model_class, file_path):
    """
    Loads the model from disk.
    TODO: Add loading with params
    """
    model = model_class()
    model.load_state_dict(torch.load(file_path))
    return model


class MLP(nn.Module):
    """
    Basic MLP component.
    """
    def __init__(self, input_size, hidden_sizes=[256,256], output_size=32, final_activation=None, gain=1):
        super(MLP, self).__init__()

        # To keep all hidden layers
        self.fc_layers = nn.ModuleList()

        # Input to hidden layers
        self.fc_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.fc_layers.append(nn.ReLU())
        
        # Between hidden layers
        for i in range(1, len(hidden_sizes)):
            self.fc_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.fc_layers.append(nn.ReLU())

        # Hidden to output
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], output_size))

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