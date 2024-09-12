"""The basics module houses the private methods generally used for model definitions.

Although all methods are intended to be private, we provide documentation
for those who would like to construct their own models using these helpers.
"""

from typing import Literal

import torch
import torch.nn as nn
from pyro.distributions.util import broadcast_shape
from torch.nn.functional import softmax, softplus


def _split_in_half(t):
    """Function to split a tensor in half.

    Courtesy of:  https://pyro.ai/examples/scanvi.html

    Parameters
    ----------
    t : torch.Tensor
        Tensor to be split

    Returns
    -------
    ts : tuple
        Size 2 tuple of tensors that are the halves of the original `torch.Tensor`.
    """
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


def _broadcast_inputs(input_args):
    """
    Helper for broadcasting shapes.

    Courtesy of:  https://pyro.ai/examples/scanvi.html

    Parameters
    ----------
    input_args : array-like
        Array-like of `torch.Tensor` to broadcast.

    Returns
    -------
    input_args : array-like
        Array-like of `torch.Tensor` that includes broadcasted tensors.
    """
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args


def _make_fc(dims):
    """
    Helper to make FC layers in quick succession for hidden layers.

    Courtesy of:  https://pyro.ai/examples/scanvi.html

    Parameters
    ----------
    dims : array-like
        Array-like of `int` specifying the sizes for layers. `dims[0], dims[-1]` are input and output respectively.

    Returns
    -------
    layers : nn.Sequential
        The layers packed into a single module.
    """
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:], strict=False):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])


# Helper to make functions between variables
class _make_func(nn.Module):
    """
    Helper to make functions for variational posteriors. Inherits `nn.Module`

    Wraps around `_make_fc` for various distribution configurations
    to reduce redundancy when defining the actual models.

    Parameters
    ----------
    in_dims : int
        Size of the input layer.

    hidden_dims : array_like
        1D Array-like of `int`. Includes sizes for intermediate layers.

    out_dim : int
        Size of the output layer.

    last_config : {"default", "+lognormal", "reparam"}, default: "default"
        The parameterization that is expected by `dist_config`.

    dist_config : {"normal", "zinb", "categorical", "+lognormal", "classifier"}, default: "normal"
        The distribution for the parameter that corresponds to the modelled layer.

    Notes
    -----
    The value for `dist_config` inherently determines the value for `last_config`, but they
    should be manually provided specifically. This helps readability in model definitions to
    make sure that the parameterizations are correct.
    """

    ## Forwards for different configs
    def _zinb_forward(self, inputs):
        gate_logits, mu = _split_in_half(self.fc(inputs))
        mu = softmax(mu, dim=-1)

        return gate_logits, mu

    def _classifier_forward(self, inputs):
        logits = self.fc(inputs)
        return logits

    def _normal_forward(self, inputs):
        ## Pre-conditions below

        _inputs = inputs.reshape(-1, inputs.size(-1))
        hidden = self.fc(_inputs)
        hidden = hidden.reshape(inputs.shape[:-1] + hidden.shape[-1:])

        loc, scale = _split_in_half(hidden)
        scale = softplus(scale)

        return loc, scale

    def _nl_forward(self, inputs):
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
                f_func = self._zinb_forward

            case "classifier":  # For discriminators
                f_func = self._classifier_forward

            case "normal":  # For count precursors
                f_func = self._normal_forward

            case "+lognormal":  # For counts
                f_func = self._nl_forward

        self.fc = _make_fc(dims)
        self.forward = f_func
