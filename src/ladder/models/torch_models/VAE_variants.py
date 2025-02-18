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

    @staticmethod
    def _get_input_args(*args):
        return args

    @staticmethod
    def _get_latent_args(*args):
        return args

    @staticmethod
    def _get_output_args(*args):
        return args

    def save(self, path):
        """Saves model parameters to disk.

        Parameters
        ----------
        path : `str`
            Path to save model parameters.
        """
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")

    def load(self, path, map_location=None):
        """Loads model parameters from disk.

        Parameters
        ----------
        path : `str`
            Path to find model parameters. Should not include the extensions `_torch.pth` or `_pyro.pth`.

        map_location : `str`, default: None
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


class GaussianVAEMixin(VAEMixin):
    """Includes shared capabilities for GaussianVAE based models.

    Methods
    -------
    model(*args)
        Generative model for the gaussian VAE.

    guide(*args)
        Approximate variational posterior for the gaussian VAE.

    """

    def model(self, *args):
        """Generative model for the base VAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module("GaussianVAE", self)

        with (
            pyro.plate("batch", x.shape[0]),
            pyro.poutine.scale(scale=1.0 / x.shape[0]),
        ):

            z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
                x.device
            ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            x_loc, x_scale = self.decoder(self._get_latent_args(z, *args))

            pyro.sample(
                "obs",
                dist.Normal(x_loc, x_scale).to_event(1),
                obs=self._get_output_args(*args),
            )

    def guide(self, *args):
        """Approximate variational posterior for the base VAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module("GaussianVAE", self)

        with (
            pyro.plate("batch", x.shape[0]),
            pyro.poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z_loc, z_scale = self.encoder(x)

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


class GaussianVAE(GaussianVAEMixin, nn.Module):
    """Base VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    hidden_dim : `int` or array_like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array_like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`.


    Methods
    -------
    __init__(in_dim, hidden_dim=128, num_layers=2, latent_dim=10)
        Constructor for the base VAE.

    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
    ):
        nn.Module.__init__(self)
        self.latent_dim, self.in_dim = latent_dim, in_dim

        self.encoder = GaussianMLP(in_dim, [hidden_dim] * num_layers, self.latent_dim)
        self.decoder = GaussianMLP(self.latent_dim, [hidden_dim] * num_layers, in_dim)

    @staticmethod
    def _get_input_args(*args):
        return args[0]

    @staticmethod
    def _get_latent_args(z, *args):
        return z

    @staticmethod
    def _get_output_args(*args):
        return args[0]


class GaussianCVAE(GaussianVAEMixin, nn.Module):
    """Conditional VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    labels_dim : `int`
        Size of the labels.

    hidden_dim : `int` or array_like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array_like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`.


    Methods
    -------
    __init__(in_dim, labels_dim, hidden_dim=128, num_layers=2, latent_dim=10)
        Constructor for the CVAE.

    """

    def __init__(
        self,
        in_dim: int,
        label_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
    ):
        nn.Module.__init__(self)
        self.latent_dim = latent_dim

        self.encoder = GaussianMLP(
            in_dim + label_dim, [hidden_dim] * num_layers, self.latent_dim
        )
        self.decoder = GaussianMLP(
            self.latent_dim + label_dim, [hidden_dim] * num_layers, in_dim
        )

    @staticmethod
    def _get_input_args(*args):
        return torch.concatenate(args[:2], dim=-1)

    @staticmethod
    def _get_latent_args(z, *args):
        return torch.concatenate((z, args[1]), dim=-1)

    @staticmethod
    def _get_output_args(*args):
        return args[0]
