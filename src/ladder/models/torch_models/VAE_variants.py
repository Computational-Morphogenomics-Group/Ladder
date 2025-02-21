"""The VAE variants module houses base model definitions for VAE style models."""

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from torch import nn

from .MLP_variants import MLP, GaussianMLP
from .VAE_mixins import (
    _AdversarialMixin,
    _CCVAEMixin,
    _HCCVAEMixin,
    _MRVAEMixin,
    _VAEMixin,
)


class GaussianVAE(_VAEMixin, nn.Module):
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

    recon_weight : `float`, default: 20.0
        Weight of the reconstruction loss for the VAE.

    kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the VAE.


    Methods
    -------
    __init__(in_dim, hidden_dim=128, num_layers=2, latent_dim=10, recon_weight=20.0, kl_weight=1.0)
        Constructor for the base VAE.

    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        recon_weight: float = 20.0,
        kl_weight: float = 1.0,
    ):
        nn.Module.__init__(self)
        self.latent_dim, self.in_dim, self.recon_weight, self.kl_weight = (
            latent_dim,
            in_dim,
            recon_weight,
            kl_weight,
        )

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

    def _reconstruct(self, z, *args):
        return dist.Normal(*self.decoder(self._get_latent_args(z, *args)))


class GaussianCVAE(GaussianVAE, nn.Module):
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

    recon_weight : `float`, default: 20.0
        Weight of the reconstruction loss for the CVAE.

    kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the CVAE.


    Methods
    -------
    __init__(in_dim, labels_dim, hidden_dim=128, num_layers=2, latent_dim=10, recon_weight=20.0, kl_weight=1.0)
        Constructor for the CVAE.

    """

    def __init__(
        self,
        in_dim: int,
        label_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        recon_weight: float = 20.0,
        kl_weight: float = 1.0,
    ):
        GaussianVAE.__init__(
            self, in_dim, hidden_dim, num_layers, latent_dim, recon_weight, kl_weight
        )

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


class GaussianCCVAE(_CCVAEMixin, nn.Module):
    """Continuous Conditional VAE class.

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

    recon_weight : `float`, default: 20.
        Weight of the reconstruction loss for the CSVAE.

    z_kl_weight : `float`, default: 0.2
        Weight of the KL divergence loss for the common latent variable of the CCVAE.

    w_kl_weight : `float`, default: 1.
        Weight of the KL divergence loss for the conditional latent variable of the CCVAE.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10, w_dim=2, w_locs=None, w_scale=None, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0)
        Constructor for the CCVAE.

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
        w_scales: list = None,
        recon_weight: float = 20.0,
        z_kl_weight: float = 0.2,
        w_kl_weight: float = 1.0,
    ):
        nn.Module.__init__(self)
        (
            self.latent_dim,
            self.in_dim,
            self.w_dim,
            self.label_dims,
            self.recon_weight,
            self.z_kl_weight,
            self.w_kl_weight,
        ) = (
            latent_dim,
            in_dim,
            w_dim,
            label_dims,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.encoder = GaussianMLP(
            self.in_dim + sum(self.label_dims),
            [hidden_dim] * num_layers,
            self.latent_dim + sum(self.label_dims) * self.w_dim,
        )
        self.decoder = GaussianMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

        if w_locs is None:
            w_locs = [0.0, 3.0]
        if w_scales is None:
            w_scales = [0.1, 1.0]
        self.w_locs, self.w_scales = w_locs, w_scales

    def _reconstruct(self, zw):
        return dist.Normal(*self.decoder(zw))

    def model(self, *args):
        """Generative model for the CCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _CCVAEMixin.model(self, *args)

    def guide(self, *args):
        """Approximate variational posterior for the CCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _CCVAEMixin.guide(self, *args)


class GaussianCSVAE(GaussianCCVAE, _AdversarialMixin, nn.Module):
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

    w_locs : `list` of `float`, default: [0., 3.0]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.0]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.

    recon_weight : `float`, default: 20.0
        Weight of the reconstruction loss for the CSVAE.

    z_kl_weight : `float`, default: 0.2
        Weight of the KL divergence loss for the common latent variable of the CSVAE.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable of the CSVAE.

    adversarial_weight : `float`, default: 1.0
        Weight of the adversarial loss.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10,  w_dim=2, w_locs=None, w_scale=None, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0, adversarial_weight=1.0)
        Constructor for the CSVAE.

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
        w_scales: list = None,
        recon_weight: float = 20.0,
        z_kl_weight: float = 0.2,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
    ):
        GaussianCCVAE.__init__(
            self,
            in_dim,
            label_dims,
            hidden_dim,
            num_layers,
            latent_dim,
            w_dim,
            w_locs,
            w_scales,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.adversarial_weight = adversarial_weight

        for i in range(len(self.label_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.latent_dim, [hidden_dim] * num_layers, self.label_dims[i]),
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
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _CCVAEMixin.model(self, *args)

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

    def guide(self, *args):
        """Approximate variational posterior for the CSVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z = _CCVAEMixin.guide(self, *args)

            with poutine.scale(None, self.adversarial_weight):
                pyro.factor(
                    "adversarial_loss",
                    self._adversarial_from_encodings(z, self._get_label_args(*args)),
                    has_rsample=True,
                )


class GaussianHCCVAE(_HCCVAEMixin, nn.Module):
    """Hierarchical Continuous Conditional VAE class.

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
        Size of the latent variable `w`, assumed to be correlated with condition labels.

    w_locs : `list` of `float`, default: [0., 3.0]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.0]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.

    recon_weight : `float`, default: 20.0
        Weight of the reconstruction loss for the HCCVAE.

    z_kl_weight : `float`, default: 0.2
        Weight of the KL divergence loss for the common latent variable of the HCCVAE.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable of the HCCVAE.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10,  w_dim=10, , w_locs=None, w_scale=None, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0)
        Constructor for the HCCVAE.

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
        w_scales: list = None,
        recon_weight: float = 20.0,
        z_kl_weight: float = 0.2,
        w_kl_weight: float = 1.0,
    ):
        nn.Module.__init__(self)
        (
            self.latent_dim,
            self.w_dim,
            self.label_dims,
            self.recon_weight,
            self.z_kl_weight,
            self.w_kl_weight,
        ) = (
            latent_dim,
            w_dim,
            label_dims,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.rho_dim = self.latent_dim + sum(self.label_dims) * self.w_dim

        self.encoder_rho = GaussianMLP(
            in_dim,
            [hidden_dim] * num_layers,
            self.rho_dim,
        )
        self.encoder = self.encoder_z = GaussianMLP(
            self.rho_dim,
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
        self.encoder_w = GaussianMLP(
            self.rho_dim + sum(self.label_dims),
            [hidden_dim] * num_layers,
            sum(self.label_dims) * self.w_dim,
        )
        self.decoder_rho = GaussianMLP(
            self.rho_dim,
            [hidden_dim] * num_layers,
            self.rho_dim,
        )
        self.decoder = GaussianMLP(
            self.rho_dim,
            [hidden_dim] * num_layers,
            in_dim,
        )

        if w_locs is None:
            w_locs = [0.0, 3.0]
        if w_scales is None:
            w_scales = [0.1, 1.0]
        self.w_locs, self.w_scales = w_locs, w_scales

    def model(self, *args):
        """Generative model for the HCCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _HCCVAEMixin.model(self, *args)

    def guide(self, *args):
        """Approximate variational posterior for the HCCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _HCCVAEMixin.guide(self, *args)

    def _reconstruct(self, rho):
        return dist.Normal(*self.decoder(rho))


class GaussianHCSVAE(GaussianHCCVAE, _AdversarialMixin, nn.Module):
    """Hierarchical Conditional Subspace VAE class.

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
        Size of the latent variable `w`, assumed to be correlated with condition labels.

    w_locs : `list` of `float`, default: [0., 3.0]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.0]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.

    recon_weight : `float`, default: 20.0
        Weight of the reconstruction loss for the HCCVAE.

    z_kl_weight : `float`, default: 0.2
        Weight of the KL divergence loss for the common latent variable of the HCSVAE.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable of the HCSVAE.

    adversarial_weight : `float`, default: 1.0
        Weight of the adversarial loss term.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10, w_dim=10, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0, adversarial_weight=1.0)
        Constructor for the HCSVAE.

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
        w_scales: list = None,
        recon_weight: float = 20.0,
        z_kl_weight: float = 0.2,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
    ):
        GaussianHCCVAE.__init__(
            self,
            in_dim,
            label_dims,
            hidden_dim,
            num_layers,
            latent_dim,
            w_dim,
            w_locs,
            w_scales,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.adversarial_weight = adversarial_weight

        for i in range(len(self.label_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.latent_dim, [hidden_dim] * num_layers, self.label_dims[i]),
            )

    def model(self, *args):
        """Generative model for the HCSVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _HCCVAEMixin.model(self, *args)

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

    def guide(self, *args):
        """Approximate variational posterior for the HCSVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z = _HCCVAEMixin.guide(self, *args)

            with poutine.scale(None, self.adversarial_weight):
                pyro.factor(
                    "adversarial_loss",
                    self._adversarial_from_encodings(z, self._get_label_args(*args)),
                    has_rsample=True,
                )

    def classification(self, *args):
        """Calculates classifier / adversarial loss from inputs."""
        x = self._get_input_args(*args)

        rho_loc, rho_scale = self.encoder_rho(x)
        rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

        z_loc, z_scale = self.encoder_z(rho)

        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        return -1 * self._adversarial_from_encodings(z, self._get_label_args(*args))


class GaussianMRVAE(_MRVAEMixin, nn.Module):
    """Multi Resolution VAE class.

     Motivation: https://www.biorxiv.org/content/10.1101/2022.10.04.510898v2

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    labels_dim : `int`
        Size of the labels.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 10
        Size of the latent variable `w`, assumed to be correlated with condition labels.

    recon_weight : `float`, default: 20.0
        Weight of the reconstruction loss for the MRVAE.

    z_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the common latent variable of the MRVAE.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable of the MRVAE.


    Methods
    -------
    __init__(in_dim, labels_dim, hidden_dim=128, num_layers=2, latent_dim=10,  w_dim=10, recon_weight=20.0, z_kl_weight=1.0, w_kl_weight=1.0)
        Constructor for the MRVAE.

    """

    def __init__(
        self,
        in_dim: int,
        labels_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        recon_weight: float = 20.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
    ):
        nn.Module.__init__(self)
        (
            self.latent_dim,
            self.labels_dim,
            self.recon_weight,
            self.z_kl_weight,
            self.w_kl_weight,
        ) = (
            latent_dim,
            labels_dim,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.encoder = GaussianMLP(
            in_dim,
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
        self.encoder_w = GaussianMLP(
            self.latent_dim + self.labels_dim,
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
        self.decoder = GaussianMLP(
            self.latent_dim,
            [hidden_dim] * num_layers,
            in_dim,
        )

    def model(self, *args):
        """Generative model for the HCCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _MRVAEMixin.model(self, *args)

    def guide(self, *args):
        """Approximate variational posterior for the HCCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _MRVAEMixin.guide(self, *args)

    def _reconstruct(self, w):
        return dist.Normal(*self.decoder(w))
