"""The VAE variants module houses base model definitions for VAE style models."""

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn

from .MLP_variants import MLP, GaussianMLP


class _ModelMixin:
    """Includes shared capabilities for models.

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


class _GaussianVAEMixin(_ModelMixin):
    """Includes shared capabilities for GaussianVAE based models.

    Methods
    -------
    model(*args)
        Generative model for the gaussian VAE.

    guide(*args)
        Approximate variational posterior for the gaussian VAE.

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

    def model(self, *args):
        """Generative model for the base VAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

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

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            pyro.poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z_loc, z_scale = self.encoder(x)

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


class GaussianVAE(_GaussianVAEMixin, nn.Module):
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


class GaussianCVAE(_GaussianVAEMixin, nn.Module):
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


class GaussianCSVAE(_GaussianVAEMixin, nn.Module):
    """Conditional Subspace VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 2
        Size of the latent variable `w`, assumed to be correlated with condition labels

    w_locs : `list` of `float`, default: [0., 3.]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.




    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10)
        Constructor for the CSVAE.

    adversarial(*args)
        Calculates classifier / adversarial loss from inputs.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None[0.1, 1.0],
    ):
        nn.Module.__init__(self)
        self.latent_dim, self.w_dim, self.label_dims = latent_dim, w_dim, label_dims

        self.encoder = GaussianMLP(
            in_dim + sum(self.label_dims),
            [hidden_dim] * num_layers,
            self.latent_dim + sum(self.label_dims) * self.w_dim,
        )
        self.decoder = GaussianMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            in_dim,
        )

        for i in range(len(self.label_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.latent_dim, [hidden_dim] * num_layers, self.label_dims[i]),
            )

        if w_locs is None:
            w_locs = [0.0, 3.0]
        if w_scales is None:
            w_scales = [0.1, 1.0]
        self.w_locs, self.w_scales = w_locs, w_scales

    @staticmethod
    def _get_input_args(*args):
        return torch.concatenate(args[:2], dim=-1)

    @staticmethod
    def _get_output_args(*args):
        return args[0]

    @staticmethod
    def _get_label_args(*args):
        return args[1]

    @staticmethod
    def _concat_lat_dims(labels, ref_list, dim):
        idxs = labels.int()
        return (
            torch.tensor(
                np.array(
                    [
                        np.concatenate([[ref_list[num]] * dim for num in elem])
                        for elem in idxs
                    ]
                )
            )
            .type_as(labels)
            .to(labels.device)
        )

    def model(self, *args):
        """Generative model for the CSVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            pyro.poutine.scale(scale=1.0 / x.shape[0]),
        ):

            z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
                x.device
            ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

            ys, attr_track = [], 0

            for i in pyro.plate("label_dims", len(self.label_dims)):
                ys.append(
                    pyro.sample(
                        f"y_{i}",
                        dist.OneHotCategorical(logits=x.new_zeros(self.label_dims[i])),
                        obs=y[..., attr_track : attr_track + self.label_dims[i]],
                    )
                )

                attr_track = attr_track + self.label_dims[i]

            w_loc, w_scale = torch.concat(
                [self._concat_lat_dims(y, self.w_locs, self.w_dim) for y in ys], dim=-1
            ), torch.concat(
                [self._concat_lat_dims(y, self.w_scales, self.w_dim) for y in ys],
                dim=-1,
            )

            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

            x_loc, x_scale = self.decoder(torch.concatenate((z, w), dim=-1))

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

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            pyro.poutine.scale(scale=1.0 / x.shape[0]),
        ):
            zw_loc, zw_scale = self.encoder(x)

            z_loc, z_scale, w_loc, w_scale = (
                zw_loc[..., : self.latent_dim],
                zw_scale[..., : self.latent_dim],
                zw_loc[..., self.latent_dim :],
                zw_scale[..., self.latent_dim :],
            )

            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
            pyro.factor(
                "adversarial_loss",
                self._adversarial_from_encodings(z, self._get_label_args(*args)),
                has_rsample=True,
            )

    def _adversarial_from_encodings(self, z, y):
        classification_loss_z, attr_track = 0, 0

        for i in pyro.plate("label_dims", len(self.label_dims)):
            classification_loss_z += dist.OneHotCategorical(
                logits=getattr(self, f"classifiers_{i}")(z)
            ).log_prob(y[..., attr_track : attr_track + self.label_dims[i]])

            attr_track = attr_track + self.label_dims[i]

        return classification_loss_z

    def classification(self, *args):
        """Calculates classifier / adversarial loss from inputs."""
        x = self._get_input_args(*args)

        zw_loc, zw_scale = self.encoder(x)
        z_loc, z_scale = (
            zw_loc[..., : self.latent_dim],
            zw_scale[..., : self.latent_dim],
        )

        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        return -1 * self._adversarial_from_encodings(z, self._get_label_args(*args))
