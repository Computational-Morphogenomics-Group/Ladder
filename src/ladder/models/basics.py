####################################
## Basic model components ##########
####################################

import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from pyro.distributions.util import broadcast_shape
from typing import Literal


def _split_in_half(t):
    """
    Splits a tensor in half along the final dimension

    Source: https://pyro.ai/examples/scanvi.html
    """
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


def _broadcast_inputs(input_args):
    """
    Helper for broadcasting inputs to neural net

    Source: https://pyro.ai/examples/scanvi.html
    """
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args


def _make_fc(dims):
    """
    Helper to make FC layers in succession

    Source: https://pyro.ai/examples/scanvi.html
    """
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])


# Helper to make functions between variables
class _make_func(nn.Module):
    """
    Helper to construct NN functions
    """

    ## Forwards for different configs
    def zinb_forward(self, inputs):
        gate_logits, mu = _split_in_half(self.fc(inputs))
        mu = softmax(mu, dim=-1)

        return gate_logits, mu

    def classifier_forward(self, inputs):
        logits = self.fc(inputs)
        return logits

    def normal_forward(self, inputs):

        ## Pre-conditions below

        _inputs = inputs.reshape(-1, inputs.size(-1))
        hidden = self.fc(_inputs)
        hidden = hidden.reshape(inputs.shape[:-1] + hidden.shape[-1:])

        loc, scale = _split_in_half(hidden)
        scale = softplus(scale)

        return loc, scale

    def nl_forward(self, inputs):
        inputs = torch.log(1 + inputs)
        h1, h2 = _split_in_half(self.fc(inputs))

        norm_loc, norm_scale = h1[..., :-1], softplus(h2[..., :-1])
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:])

        return norm_loc, norm_scale, l_loc, l_scale

    def __init__(
        self,
        in_dims,
        hidden_dims,
        out_dim,
        last_config: Literal["default", "+lognormal", "reparam"] = "default",
        dist_config: Literal[
            "normal", "zinb", "categorical", "+lognormal", "classifier"
        ] = "normal",
    ):
        super().__init__()

        # Layer configurations
        match last_config:

            case "default":  # Last will be 2*out for easy reparam
                dims = [in_dims] + hidden_dims + [out_dim]

            case "+lognormal":  # Last will include +2 for l_loc & l_scale
                dims = [in_dims] + hidden_dims + [2 * out_dim + 2]

            case "reparam":
                dims = [in_dims] + hidden_dims + [2 * out_dim]

        # Forward configurations
        match dist_config:
            case "zinb":  # For decoders
                f_func = self.zinb_forward

            case "classifier":  # For discriminators
                f_func = self.classifier_forward

            case "normal":  # For count precursors
                f_func = self.normal_forward

            case "+lognormal":  # For counts
                f_func = self.nl_forward

        self.fc = _make_fc(dims)
        self.forward = f_func
