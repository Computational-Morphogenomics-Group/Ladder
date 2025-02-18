"""The VAE variants module houses base model definitions for VAE style models."""

import pyro
import pyro.distributions as dist
import torch
from src.ladder.models.torch_models.MLP_variants import GaussianMLP
from torch import nn


class VAEMixin:
    """Includes shared capabilities for VAE based models.

    Methods
    -------
    save(path)
        Saves model parameters to disk.

    load(path, map_location=None)
        Loads model parameters from disk.

    """

    def save(self, path):
        """Saves model parameters to disk.

        Parameters
        ----------
        path : :class:`str`
            Path to save model parameters.
        """
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")

    def load(self, path, map_location=None):
        """Loads model parameters from disk.

        Parameters
        ----------
        path : :class:`str`
            Path to find model parameters. Should not include the extensions `_torch.pth` or `_pyro.pth`.

        map_location : :class:`str`, default: None
            Specifies where the model should be loaded. See :class:`~torch.device` for details.
        """
        pyro.clear_param_store()

        if map_location is None:
            self.load_state_dict(torch.load(path + "_torch.pth"))
            pyro.get_param_store().load(path + "_pyro.pth")

        else:
            self.load_state_dict(
                torch.load(path + "_torch.pth", map_location=map_location)
            )
            pyro.get_param_store().load(path + "_pyro.pth", map_location=map_location)


class GaussianVAE(VAEMixin, nn.Module):
    """Base VAE class.

    Parameters
    ----------
    in_dim : :class:`int`
        Size of the input / output space.

    hidden_dim : :class:`float` or array_like
        Size of the hidden layers.

    num_layers : :class:`float` or array_like
        Number of hidden layers.

    latent_dim : :class:`int`, default: 10
        Size of the latent variable `z`.


    Methods
    -------
    __init__(in_dim, hidden_dim=128, num_layers=2, latent_dim=10)
        Constructor for the base VAE.

    model(x, y=None)
        Generative model for the base VAE.

    guide(x, y=None)
        Approximate variational posterior for the base VAE.

    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
    ):
        nn.Module.__init__(self)
        self.latent_dim = latent_dim

        self.encoder = GaussianMLP(in_dim, [hidden_dim] * num_layers, self.latent_dim)
        self.decoder = GaussianMLP(self.latent_dim, [hidden_dim] * num_layers, in_dim)

    def model(self, *args):
        """Generative model for the base VAE.

        Parameters
        ----------
        *args :
            Only the first element will be used as input. Expected input batch with shape (N, in_dim).
        """
        x = args[0]

        pyro.module("GaussianVAE", self)

        with (
            pyro.plate("batch", x.shape[0]),
            pyro.poutine.scale(scale=1.0 / x.shape[0]),
        ):

            z_loc, z_scale = x.new_zeros(
                torch.Size((x.shape[0], self.latent_dim))
            ), x.new_ones(torch.Size((x.shape[0], self.latent_dim)))

            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            x_loc, x_scale = self.decoder(z)

            pyro.sample("obs", dist.Normal(x_loc, x_scale).to_event(1), obs=x)

    def guide(self, *args):
        """Approximate variational posterior for the base VAE.

        Parameters
        ----------
        *args :
            Only the first element will be used as input. Expected input batch with shape (N, in_dim).
        """
        x = args[0]

        pyro.module("GaussianVAE", self)

        with (
            pyro.plate("batch", x.shape[0]),
            pyro.poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z_loc, z_scale = self.encoder(x)

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
