"""The scvi_variants module houses the model definitions that are based on the scVI (https://www.nature.com/articles/s41592-018-0229-2) skeleton.

All model implementations are available through Pyro.
"""

from typing import Literal

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus

from .basics import _broadcast_inputs, _make_func, _split_in_half

# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================


class SCVI(nn.Module):
    r"""scVI (https://www.nature.com/articles/s41592-018-0229-2), implemented through `pyro`.

    Parameters
    ----------
    num_genes : :class:`int`
        Size of the gene space.

    l_loc : :class:`float` or array_like
        Either a single value for log-mean library size, or a 1D array-like of values if `batch_correction`.

    l_scale : :class:`float` or array_like
        Either a single value for log-variance library size, or a 1D array-like of values if `batch_correction`.

    hidden_dim : :class:`int`, default: 128
        Size of the hidden layers throughout the model.

    num_layers : :class:`int`, default: 2
        Number of hidden layers between any input and output layer.

    latent_dim : :class:`int`, default: 10
        Size of the latent variable `z`.

    scale_factor : :class:`float`, default: 1.0
        Factor used to scale and normalize the loss.

    batch_correction : :class:`bool`, default: False
        If `True`, expects batch to be appended to input and corrects for batch.

    reconstruction : :class:`Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"]`, default: "ZINB"
        The distribiution assumed to model the input data.

    Methods
    -------
    __init__(num_genes, l_loc, l_scale, hidden_dim=128, num_layers=2, latent_dim=10, scale_factor=1.0, batch_correction=False, reconstruction="ZINB")
        Constructor for scVI.

    model(x, y=None)
        Generative model for scVI.

    guide(x, y=None)
        Approximate variational posterior for scVI.

    generate(x, y_source=None, y_target=None)
        Function used post-training for Oracle-scVI to facilitate transfer between conditional labels.

    get_weights()
        Returns interpretable coefficients for latents.

    save(path="scvi_params")
        Saves model parameters to disk.

    load(path="scvi_params", map_location=None)
        Loads model parameters from disk.
    """

    def __init__(
        self,
        num_genes,
        l_loc,
        l_scale,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        scale_factor: float = 1.0,
        batch_correction: bool = False,
        reconstruction: Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"] = "ZINB",
    ):
        # Init params & hyperparams
        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.l_loc = l_loc
        self.l_scale = l_scale
        self.batch_correction = batch_correction  # Assume that batch is appended to input & latent if batch correction is applied
        self.reconstruction = reconstruction

        super().__init__()

        # Setup NN functions
        match self.reconstruction:
            case "ZINB":
                self.x_decoder = _make_func(
                    in_dims=self.latent_dim + int(self.batch_correction),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.num_genes,
                    last_config="reparam",
                    dist_config="zinb",
                )

            case "Normal":
                self.x_decoder = _make_func(
                    in_dims=self.latent_dim + int(self.batch_correction),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.num_genes,
                    last_config="reparam",
                    dist_config="normal",
                )

            case "ZINB_LD" | "Normal_LD":
                self.x_decoder = nn.Linear(
                    self.latent_dim + int(self.batch_correction),
                    self.num_genes * 2,
                    bias=False,
                )

        self.zl_encoder = _make_func(
            in_dims=self.num_genes + int(self.batch_correction),
            hidden_dims=[hidden_dim] * num_layers,
            out_dim=self.latent_dim,
            last_config="+lognormal",
            dist_config="+lognormal",
        )

        self.epsilon = 0.006

    # Model
    def model(self, x, y=None):
        """Generative model for scVI.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y : :class:`~torch.Tensor`, optional
            Not used in a meaningful way, kept for compatibility.
        """
        pyro.module("scvi", self)

        # Inverse dispersions
        theta = pyro.param(
            "inverse_dispersion",
            10.0 * x.new_ones(self.num_genes),
            constraint=constraints.positive,
        )

        # Loop for mini-batch
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample(
                "z", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1)
            )

            if "ZINB" in self.reconstruction:
                # If batch correction, pick corresponding loc scale
                if self.batch_correction:
                    l_loc, l_scale = (
                        torch.tensor(
                            self.l_loc[
                                x[..., -1].detach().clone().cpu().type(torch.int)
                            ]
                        )
                        .reshape(-1, 1)
                        .to(x.device),
                        torch.tensor(
                            self.l_scale[
                                x[..., -1].detach().clone().cpu().type(torch.int)
                            ]
                        )
                        .reshape(-1, 1)
                        .to(x.device),
                    )

                # Single size factor
                else:
                    l_loc, l_scale = self.l_loc * x.new_ones(
                        1
                    ), self.l_scale * x.new_ones(1)

                l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            # If batch corrected, use batch to go back. Else skip
            if self.batch_correction:
                z = torch.cat([z, x[..., -1].view(-1, 1)], dim=-1)

            match self.reconstruction:
                case "ZINB":
                    gate_logits, mu = self.x_decoder(z)
                    nb_logits = (l * mu + self.epsilon).log() - (
                        theta + self.epsilon
                    ).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits,
                        validate_args=False,
                    )

                case "Normal":
                    x_loc, x_scale = self.x_decoder(z)
                    x_dist = dist.Normal(x_loc, x_scale)

                case "ZINB_LD":
                    gate_logits, mu = _split_in_half(self.x_decoder(z))
                    mu = softmax(mu, dim=-1)
                    nb_logits = (l * mu + self.epsilon).log() - (
                        theta + self.epsilon
                    ).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits,
                        validate_args=False,
                    )

                case "Normal_LD":
                    _z = z.reshape(-1, z.size(-1))
                    out = self.x_decoder(_z)
                    out = out.reshape(z.shape[:-1] + out.shape[-1:])

                    x_loc, x_scale = _split_in_half(out)
                    x_scale = softplus(x_scale)
                    x_dist = dist.Normal(x_loc, x_scale)

            # If batch corrected, we expect last index to be batch
            if self.batch_correction:
                pyro.sample("x", x_dist.to_event(1), obs=x[..., :-1])
            else:
                pyro.sample("x", x_dist.to_event(1), obs=x)

    # Guide
    def guide(self, x, y=None):
        """Approximate variational posterior for scVI.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y : :class:`~torch.Tensor`, optional
            Not used in a meaningful way, kept for compatibility.
        """
        pyro.module("scvi", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # If batch corrected, this is expression appended with batch
            z_loc, z_scale, l_loc, l_scale = self.zl_encoder(x)

            if "ZINB" in self.reconstruction:
                pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    # Generate
    def generate(self, x, y_source=None, y_target=None):
        """Function used post-training for Oracle-scVI to facilitate transfer between conditional labels.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y_source : :class:`~torch.Tensor`, optional
            Not used in a meaningful way, kept for compatibility.

        y_target : :class:`~torch.Tensor`, optional
            Not used in a meaningful way, kept for compatibility.


        Returns
        -------
        x_rec : :class:`~torch.Tensor`
            Reconstructed gene counts.
        """
        pyro.module("scvi", self)

        ## Encode
        z_loc, z_scale, l_loc, l_scale = self.zl_encoder(x)

        l_enc = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
        z_enc = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        # If batch correction, then append batch to latent
        if self.batch_correction:
            z_enc = torch.cat([z_enc, x[..., -1].view(-1, 1)], dim=-1)

        match self.reconstruction:
            case "ZINB":
                gate_logits, mu = self.x_decoder(z_enc)
                nb_logits = (l_enc * mu + self.epsilon).log() - (
                    theta.to(mu.device) + self.epsilon
                ).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(
                    gate_logits=gate_logits,
                    total_count=theta,
                    logits=nb_logits,
                    validate_args=False,
                )

            case "Normal":
                x_loc, x_scale = self.x_decoder(z_enc)
                x_dist = dist.Normal(x_loc, x_scale)

            case "ZINB_LD":
                gate_logits, mu = _split_in_half(self.x_decoder(z_enc))
                mu = softmax(mu, dim=-1)
                nb_logits = (l_enc * mu + self.epsilon).log() - (
                    theta.to(mu.device) + self.epsilon
                ).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(
                    gate_logits=gate_logits,
                    total_count=theta,
                    logits=nb_logits,
                    validate_args=False,
                )

            case "Normal_LD":
                _z_enc = z_enc.reshape(-1, z_enc.size(-1))
                out = self.x_decoder(_z_enc)
                out = out.reshape(z_enc.shape[:-1] + out.shape[-1:])

                x_loc, x_scale = _split_in_half(out)
                x_scale = softplus(x_scale)
                x_dist = dist.Normal(x_loc, x_scale)

        x_rec = pyro.sample("x", x_dist.to_event(1))
        return x_rec

    # Get linear weights if LD
    def get_weights(self):
        """Returns interpretable coefficients for latents.

        Refer to Notes for details.

        Returns
        -------
        loc, mu : :class:`~torch.Tensor`
            Mu of ZINB or Gaussian.

        scale, logits : :class:`~torch.Tensor`
            Either the variance of the Gaussian or ZI logits for ZINB.
        """
        assert self.reconstruction.endswith("LD")
        match self.reconstruction:
            case "ZINB_LD":
                if self.batch_correction:
                    logits, mu = _split_in_half(
                        list(self.x_decoder.parameters())[0].T[:-1].detach().cpu()
                    )
                else:
                    logits, mu = _split_in_half(
                        list(self.x_decoder.parameters())[0].T.detach().cpu()
                    )
                return mu, logits

            case "Normal_LD":
                if self.batch_correction:
                    loc, scale = _split_in_half(
                        list(self.x_decoder.parameters())[0].T[:-1].detach().cpu()
                    )
                else:
                    loc, scale = _split_in_half(
                        list(self.x_decoder.parameters())[0].T.detach().cpu()
                    )
                return loc, scale

    # Save self
    def save(self, path="scvi_params"):
        """Saves model parameters to disk.

        Parameters
        ----------
        path : :class:`str`, default: "scvi_params"
            Path to save model parameters.
        """
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")

    # Load
    def load(self, path="scvi_params", map_location=None):
        """Loads model parameters from disk.

        Parameters
        ----------
        path : :class:`str`, default: "scvi_params"
            Path to find model parameters. Should not include the extensions `_torch.pth` or `_pyro.pth` or any such variant.

        map_location : :class:`str`, optional
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


# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================


## scANVI taken from https://pyro.ai/examples/scanvi.html
class SCANVI(nn.Module):
    """Supervised scANVI (https://www.embopress.org/doi/full/10.15252/msb.20209620), implemented through `pyro`.

    Parameters
    ----------
    num_genes : :class:`int`
        Size of the gene space.

    l_loc : :class:`float` or array_like
        Either a single value for log-mean library size, or a 1D array-like of values if `batch_correction`.

    l_scale : :class:`float` or array_like
        Either a single value for log-variance library size, or a 1D array-like of values if `batch_correction`.

    num_labels : :class:`int`
        Length of the one-hot encoded condition labels expected in the data.

    hidden_dim : :class:`int`, default: 128
        Size of the hidden layers throughout the model.

    num_layers : :class:`int`, default: 2
        Number of hidden layers between any input and output layer.

    latent_dim : :class:`int`
        Size of the latent variable `z`.

    alpha : :class:`float`
        Factor used to scale the classifier loss.

    scale_factor : :class:`float`, default: 1.0
        Factor used to scale and normalize the loss.

    batch_correction : :class:`bool`, default: False
        If `True`, expects batch to be appended to input and corrects for batch.

    reconstruction : :class:`Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"]`, default: "ZINB"
        The distribiution assumed to model the input data.

    Methods
    -------
    __init__(num_genes, l_loc, l_scale, num_labels, hidden_dim=128, num_layers=2, latent_dim=10, alpha=1.0, scale_factor=1.0, batch_correction=False, reconstruction="ZINB")
        Constructor for supervised scANVI.

    model(x, y)
        Generative model for supervised scANVI.

    guide(x, y)
        Approximate variational posterior for supervised scANVI.

    generate(x, y_source, y_target)
        Function used post-training for supervised scANVI to facilitate transfer between conditional labels.

    get_weights()
        Returns interpretable coefficients for latents.

    save(path="scanvi_params")
        Saves model parameters to disk.

    load(path="scanvi_params", map_location=None)
        Loads model parameters from disk.
    """

    def __init__(
        self,
        num_genes,
        l_loc,
        l_scale,
        num_labels,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        alpha: float = 1,
        scale_factor: float = 1.0,
        batch_correction: bool = False,
        reconstruction: Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"] = "ZINB",
    ):
        # Init params & hyperparams
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.l_loc = l_loc
        self.l_scale = l_scale
        self.batch_correction = batch_correction  # Assume that batch is appended to input & latent if batch correction is applied
        self.reconstruction = reconstruction

        super().__init__()

        # Setup NN functions

        match self.reconstruction:
            case "ZINB":
                self.z2_decoder = _make_func(
                    in_dims=self.latent_dim + self.num_labels,
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.latent_dim,
                    last_config="reparam",
                    dist_config="normal",
                )

                self.x_decoder = _make_func(
                    in_dims=self.latent_dim + int(self.batch_correction),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.num_genes,
                    last_config="reparam",
                    dist_config="zinb",
                )

            case "Normal":
                self.z2_decoder = _make_func(
                    in_dims=self.latent_dim + self.num_labels,
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.latent_dim,
                    last_config="reparam",
                    dist_config="normal",
                )

                self.x_decoder = _make_func(
                    in_dims=self.latent_dim + int(self.batch_correction),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.num_genes,
                    last_config="reparam",
                    dist_config="normal",
                )

            case "ZINB_LD" | "Normal_LD":
                self.x_decoder = nn.Linear(
                    self.latent_dim + self.num_labels + int(self.batch_correction),
                    self.num_genes * 2,
                    bias=False,
                )

        self.z2l_encoder = _make_func(
            in_dims=self.num_genes + int(self.batch_correction),
            hidden_dims=[hidden_dim] * num_layers,
            out_dim=self.latent_dim,
            last_config="+lognormal",
            dist_config="+lognormal",
        )

        self.classifier = _make_func(
            in_dims=self.latent_dim,
            hidden_dims=[hidden_dim] * num_layers,
            out_dim=self.num_labels,
            last_config="default",
            dist_config="classifier",
        )

        self.z1_encoder = _make_func(
            in_dims=self.num_labels + self.latent_dim,
            hidden_dims=[hidden_dim] * num_layers,
            out_dim=self.latent_dim,
            last_config="reparam",
            dist_config="normal",
        )

        self.epsilon = 0.006

    # Model
    def model(self, x, y):
        """Generative model for scANVI.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y : :class:`~torch.Tensor`
            One-hot encoded conditional labels.
        """
        pyro.module("scanvi", self)

        # Inverse dispersions
        theta = pyro.param(
            "inverse_dispersion",
            10.0 * x.new_ones(self.num_genes),
            constraint=constraints.positive,
        )

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample(
                "z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1)
            )
            y = pyro.sample(
                "y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y
            )

            z1_y = torch.cat([z1, y], dim=-1)

            if "ZINB" in self.reconstruction:
                # If batch correction, pick corresponding loc scale
                if self.batch_correction:
                    l_loc, l_scale = (
                        torch.tensor(
                            self.l_loc[
                                x[..., -1].detach().clone().cpu().type(torch.int)
                            ]
                        )
                        .reshape(-1, 1)
                        .to(x.device),
                        torch.tensor(
                            self.l_scale[
                                x[..., -1].detach().clone().cpu().type(torch.int)
                            ]
                        )
                        .reshape(-1, 1)
                        .to(x.device),
                    )

                # Single size factor
                else:
                    l_loc, l_scale = self.l_loc * x.new_ones(
                        1
                    ), self.l_scale * x.new_ones(1)

                l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            match self.reconstruction:
                case "ZINB":
                    z2_loc, z2_scale = self.z2_decoder(z1_y)
                    z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

                    # Append batch here if corrected
                    if self.batch_correction:
                        z2 = torch.cat([z2, x[..., -1].view(-1, 1)], dim=-1)

                    gate_logits, mu = self.x_decoder(z2)
                    nb_logits = (l * mu + self.epsilon).log() - (
                        theta + self.epsilon
                    ).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits,
                        validate_args=False,
                    )

                case "Normal":
                    z2_loc, z2_scale = self.z2_decoder(z1_y)
                    z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

                    # Append batch here if corrected
                    if self.batch_correction:
                        z2 = torch.cat([z2, x[..., -1].view(-1, 1)], dim=-1)

                    x_loc, x_scale = self.x_decoder(z2)
                    x_dist = dist.Normal(x_loc, x_scale)

                case "ZINB_LD":
                    # Append the batch
                    if self.batch_correction:
                        z1_y = torch.cat([z1_y, x[..., -1].view(-1, 1)], dim=-1)

                    gate_logits, mu = _split_in_half(self.x_decoder(z1_y))
                    mu = softmax(mu, dim=-1)
                    nb_logits = (l * mu + self.epsilon).log() - (
                        theta + self.epsilon
                    ).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits,
                        validate_args=False,
                    )

                case "Normal_LD":
                    # Append the batch
                    if self.batch_correction:
                        z1_y = torch.cat([z1_y, x[..., -1].view(-1, 1)], dim=-1)

                    _z1_y = z1_y.reshape(-1, z1_y.size(-1))
                    out = self.x_decoder(_z1_y)
                    out = out.reshape(z1_y.shape[:-1] + out.shape[-1:])

                    x_loc, x_scale = _split_in_half(out)
                    x_scale = softplus(x_scale)
                    x_dist = dist.Normal(x_loc, x_scale)

            # If batch corrected, we expect last index to be batch
            if self.batch_correction:
                pyro.sample("x", x_dist.to_event(1), obs=x[..., :-1])
            else:
                pyro.sample("x", x_dist.to_event(1), obs=x)

    # Guide
    def guide(self, x, y):
        """Approximate variational posterior for scANVI.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y : :class:`~torch.Tensor`
            One-hot encoded conditional labels.
        """
        pyro.module("scanvi", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)

            if "ZINB" in self.reconstruction:
                pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            if self.reconstruction in ["ZINB_LD", "Normal_LD"]:
                z2 = pyro.sample(
                    "z2",
                    dist.Normal(z2_loc, z2_scale).to_event(1),
                    infer={"is_auxiliary": True},
                )

            else:
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            classification_loss = y_dist.log_prob(y)
            pyro.factor(
                "classification_loss",
                -self.alpha * classification_loss,
                has_rsample=False,
            )

            z2_y = _broadcast_inputs([z2, y])
            z2_y = torch.cat(z2_y, dim=-1)
            z1_loc, z1_scale = self.z1_encoder(z2_y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

    # Function to move points between conditions
    @torch.no_grad()
    def generate(self, x, y_source, y_target):
        """Function used post-training for Supervies-scANVI to facilitate transfer between conditional labels.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y_source : :class:`~torch.Tensor`
            One-hot encoded conditional labels for the input.

        y_target : :class:`~torch.Tensor`
            One-hot encoded conditional labels for the targets. Must be the same size in the first dimension as input.
        """
        pyro.module("scanvi", self)

        ## Encode
        # Variational for rho & l
        z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)

        l_enc = pyro.sample("l_enc", dist.LogNormal(l_loc, l_scale).to_event(1))
        z2_enc = pyro.sample("z2_enc", dist.Normal(z2_loc, z2_scale).to_event(1))

        # Variational for z
        z2_y = _broadcast_inputs([z2_enc, y_source])
        z2_y = torch.cat(z2_y, dim=-1)
        z1_loc, z1_scale = self.z1_encoder(z2_y)
        z1_enc = pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        z1_y = torch.cat([z1_enc, y_target], dim=-1)

        match self.reconstruction:
            case "ZINB":
                z2_loc, z2_scale = self.z2_decoder(z1_y)
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

                if self.batch_correction:
                    z2 = torch.cat([z2, x[..., -1].view(-1, 1)], dim=-1)

                gate_logits, mu = self.x_decoder(z2)
                nb_logits = (l_enc * mu + self.epsilon).log() - (
                    theta.to(mu.device) + self.epsilon
                ).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(
                    gate_logits=gate_logits,
                    total_count=theta,
                    logits=nb_logits,
                    validate_args=False,
                )

            case "Normal":
                z2_loc, z2_scale = self.z2_decoder(z1_y)
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

                if self.batch_correction:
                    z2 = torch.cat([z2, x[..., -1].view(-1, 1)], dim=-1)

                x_loc, x_scale = self.x_decoder(z2)
                x_dist = dist.Normal(x_loc, x_scale)

            case "ZINB_LD":
                # Append the batch
                if self.batch_correction:
                    z1_y = torch.cat([z1_y, x[..., -1].view(-1, 1)], dim=-1)

                gate_logits, mu = _split_in_half(self.x_decoder(z1_y))
                mu = softmax(mu, dim=-1)
                nb_logits = (l_enc * mu + self.epsilon).log() - (
                    theta.to(mu.device) + self.epsilon
                ).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(
                    gate_logits=gate_logits,
                    total_count=theta,
                    logits=nb_logits,
                    validate_args=False,
                )

            case "Normal_LD":
                # Append the batch
                if self.batch_correction:
                    z1_y = torch.cat([z1_y, x[..., -1].view(-1, 1)], dim=-1)

                _z1_y = z1_y.reshape(-1, z1_y.size(-1))
                out = self.x_decoder(_z1_y)
                out = out.reshape(z1_y.shape[:-1] + out.shape[-1:])

                x_loc, x_scale = _split_in_half(out)
                x_scale = softplus(x_scale)
                x_dist = dist.Normal(x_loc, x_scale)

        # Observe the datapoint x using the observation distribution x_dist
        x_rec = pyro.sample("x", x_dist.to_event(1))

        return x_rec

    def get_weights(self):
        """Returns interpretable coefficients for latents.

        Refer to Notes for details.


        Returns
        -------
        loc, mu : :class:`~torch.Tensor`
            Mu of ZINB or Gaussian.

        scale, logits : :class:`~torch.Tensor`
            Either the variance of the Gaussian or ZI logits for ZINB.
        """
        assert self.reconstruction.endswith("LD")
        match self.reconstruction:
            case "ZINB_LD":
                if self.batch_correction:
                    logits, mu = _split_in_half(
                        list(self.x_decoder.parameters())[0].T[:-1].detach().cpu()
                    )
                else:
                    logits, mu = _split_in_half(
                        list(self.x_decoder.parameters())[0].T.detach().cpu()
                    )
                return mu, logits

            case "Normal_LD":
                if self.batch_correction:
                    loc, scale = _split_in_half(
                        list(self.x_decoder.parameters())[0].T[:-1].detach().cpu()
                    )
                else:
                    loc, scale = _split_in_half(
                        list(self.x_decoder.parameters())[0].T.detach().cpu()
                    )
                return loc, scale

    # Save self
    def save(self, path="scanvi_params"):
        """Saves model parameters to disk.

        Parameters
        ----------
        path : :class:`str`, default: "scanvi_params"
            Path to save model parameters.
        """
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")

    # Load
    def load(self, path="scanvi_params", map_location=None):
        """Loads model parameters from disk.

        Parameters
        ----------
        path : :class:`str`, default: "scanvi_params"
            Path to find model parameters. Should not include the extensions `_torch.pth` or `_pyro.pth` or any such variant.

        map_location : :class:`str`, optional
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


# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================


class Patches(nn.Module):
    """Patches, the multi-attribute model.

    Parameters
    ----------
    num_genes : :class:`int`
        Size of the gene space.

    l_loc : :class:`float` or array_like
        Either a single value for log-mean library size, or a 1D array-like of values if `batch_correction`.

    l_scale : :class:`float` or array_like
        Either a single value for log-variance library size, or a 1D array-like of values if `batch_correction`.

    num_labels : :class:`int`
        Total number of attributes present in the data.

    len_attrs : array_like
        1D Array-like of `int`. Specifies how many attributes per condition class.

    betas : array_like, optional
        Scales the adversarial & classifier loss for each condition class.

    w_loc : array_like, optional
        Means for the conditionally selected multivariate gaussians to parameterize each `w_k`.

    w_scale : array_like, optional
        Stds for the conditionally selected multivariate gaussians to parameterize each `w_k`.

    w_dim : :class:`int`, default: 2
        Size of each `w_k` for the attributes.

    w_kl : :class:`float`, default: 1.0
        KL weight for conditional latents.

    latent_dim : :class:`int`
        Size of the latent variable `z`.

    z_kl : :class:`float`, default: 1.0
        KL weight for common latent.

    recon_weight : :class:`float`, default: 1.0
        Weight for reconstructin error.

    num_layers : :class:`int`, default: 2
        Number of hidden layers between any input and output layer.

    hidden_dim : :class:`int`, default: 128
        Size of the hidden layers throughout the model.

    scale_factor : :class:`float`, default: 1.0
        Factor used to scale and normalize the loss.

    batch_correction : :class:`bool`, default: False
        If `True`, expects batch to be appended to input and corrects for batch.

    ld_sparsity : :class:`bool`, default: False
        If `True`, adds L1 loss for attribute specific latents. Can only be used with linear decoders.

    ld_normalize : :class:`bool`, default: True
        If `True`, adds bias term to decoder. Can only be used with linear decoders.

    sparsity_lambda : :class:`float`, default: 0.0001
        Weight of the L1 term in the loss.

    reconstruction : :class:`Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"]`, default: "ZINB"
        The distribiution assumed to model the input data.

    Methods
    -------
    __init__(num_genes, l_loc, l_scale, num_labels, len_attrs, betas=None, w_loc=None, w_scale=None, w_dim=2, latent_dim=10, num_layers=2, hidden_dim=128, scale_factor=1.0, batch_correction=False, ld_sparsity=False, ld_normalize=True, reconstruction="ZINB")
        Constructor for Patches.

    model(x, y)
        Generative model for Patches.

    guide(x, y)
        Approximate variational posterior for Patches.

    adversarial(x,y)
        Adversarial loss for Patches.

    generate(x, y_source, y_target)
        Function used post-training for Patches to transfer between conditional labels.

    get_weights()
        Returns interpretable coefficients for latents.

    save(path="scanvi_params")
        Saves model parameters to disk.

    load(path="scanvi_params", map_location=None)
        Loads model parameters from disk.
    """

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

    def __init__(
        self,
        num_genes,
        l_loc,
        l_scale,
        num_labels,
        len_attrs,
        betas=None,
        w_loc=None,
        w_scale=None,
        w_dim: int = 2,
        w_kl: float = 1.0,
        latent_dim: int = 10,
        z_kl: float = 1.0,
        recon_weight: float = 1.0,
        num_layers: int = 2,
        hidden_dim: int = 128,
        scale_factor: float = 1.0,
        batch_correction: bool = False,
        ld_sparsity: bool = False,
        ld_normalize: bool = False,
        sparsity_lambda: float = 0.0001,
        reconstruction: Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"] = "ZINB",
    ):
        # Init params & hyperparams
        self.len_attrs = (
            len_attrs  # List keeping number of possibilities for each attribute
        )

        # Handle betas
        if betas is None:
            self.betas = [1] * len(self.len_attrs)
        else:
            assert len(betas) == len(self.len_attrs)
            self.betas = betas

        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.w_dim = w_dim  # Latent dimension for each label
        self.l_loc = l_loc
        self.l_scale = l_scale

        if w_loc is None:
            w_loc = [0, 3]

        self.w_locs = w_loc  # Prior means for attribute being 0,1 (indices correspond to attribute value)

        if w_scale is None:
            w_scale = [0.1, 1]

        self.w_scales = w_scale  # Prior scales for attribute being 0,1 (indices correspond to attribute value)

        self.w_kl = w_kl  # KL weight for conditionals
        self.z_kl = z_kl  # KL weight for common
        self.recon_weight = recon_weight  # Weight for recon
        self.batch_correction = batch_correction  # Assume that batch is appended to input & latent if batch correction is applied
        self.reconstruction = reconstruction  # Distribution for the reconstruction
        self.sparsity = ld_sparsity  # Sparsity, used only with LD
        self.sparsity_lambda = sparsity_lambda  # Sparsity lambda, used only with LD
        self.normalize = ld_normalize  # Normalization, adds bias to LD
        self.epsilon = 0.006

        super().__init__()

        # Setup NN functions

        match self.reconstruction:
            case "ZINB":
                self.rho_decoder = _make_func(
                    in_dims=self.latent_dim + (self.w_dim * self.num_labels),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.latent_dim + (self.w_dim * self.num_labels),
                    last_config="reparam",
                    dist_config="normal",
                )

                self.x_decoder = _make_func(
                    in_dims=self.latent_dim
                    + (self.w_dim * self.num_labels)
                    + int(self.batch_correction),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.num_genes,
                    last_config="reparam",
                    dist_config="zinb",
                )

            case "Normal":
                self.rho_decoder = _make_func(
                    in_dims=self.latent_dim + (self.w_dim * self.num_labels),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.latent_dim + (self.w_dim * self.num_labels),
                    last_config="reparam",
                    dist_config="normal",
                )

                self.x_decoder = _make_func(
                    in_dims=self.latent_dim
                    + (self.w_dim * self.num_labels)
                    + int(self.batch_correction),
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.num_genes,
                    last_config="reparam",
                    dist_config="normal",
                )

            case "ZINB_LD" | "Normal_LD":
                self.x_decoder = nn.Linear(
                    self.latent_dim
                    + (self.w_dim * self.num_labels)
                    + int(self.batch_correction),
                    self.num_genes * 2,
                    bias=self.normalize,
                )

        self.rho_l_encoder = _make_func(
            in_dims=self.num_genes + int(self.batch_correction),
            hidden_dims=[hidden_dim] * num_layers,
            out_dim=self.latent_dim + (self.w_dim * self.num_labels),
            last_config="+lognormal",
            dist_config="+lognormal",
            keep_last_batch_norm=False,
        )

        for i in range(len(self.len_attrs)):
            setattr(
                self,
                f"classifier_z_y{i}",
                _make_func(
                    in_dims=self.latent_dim,
                    hidden_dims=[hidden_dim] * num_layers,
                    out_dim=self.len_attrs[i],
                    last_config="default",
                    dist_config="classifier",
                    keep_last_batch_norm=False,
                ),
            )

        self.z_encoder = _make_func(
            in_dims=self.latent_dim + (self.w_dim * self.num_labels),
            hidden_dims=[hidden_dim] * num_layers,
            out_dim=self.latent_dim,
            last_config="reparam",
            dist_config="normal",
            keep_last_batch_norm=False,
        )
        self.w_encoder = _make_func(
            in_dims=self.latent_dim + (self.w_dim * self.num_labels) + self.num_labels,
            hidden_dims=[hidden_dim] * num_layers,
            out_dim=self.w_dim * self.num_labels,
            last_config="reparam",
            dist_config="normal",
            keep_last_batch_norm=False,
        )

    # Model
    def model(self, x, y):
        """Generative model for Patches.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y : :class:`~torch.Tensor`
            One-hot encoded and concatenated attribute labels.
        """
        pyro.module("patches", self)

        theta = pyro.param(
            "inverse_dispersion",
            10.0 * x.new_ones(self.num_genes),
            constraint=constraints.positive,
        )

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):

            with poutine.scale(None, self.z_kl):
                z = pyro.sample(
                    "z", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1)
                )

            # Keep tracked attributes in a list
            y_s = []
            attr_track = 0

            for i in pyro.plate("len_attrs", len(self.len_attrs)):
                next_track = attr_track + self.len_attrs[i]
                y_attr = pyro.sample(
                    f"y_{i}",
                    dist.OneHotCategorical(logits=x.new_zeros(self.len_attrs[i])),
                    obs=y[..., attr_track:next_track],
                )
                y_s.append(y_attr)

                attr_track = next_track

            w_loc = torch.concat(
                [self._concat_lat_dims(y, self.w_locs, self.w_dim) for y in y_s], dim=-1
            )
            w_scale = torch.concat(
                [self._concat_lat_dims(y, self.w_scales, self.w_dim) for y in y_s],
                dim=-1,
            )

            with poutine.scale(None, self.w_kl):
                w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

            zw = torch.cat([z, w], dim=-1)

            if "ZINB" in self.reconstruction:
                # If batch correction, pick corresponding loc scale
                if self.batch_correction:
                    l_loc, l_scale = (
                        torch.tensor(
                            self.l_loc[
                                x[..., -1].detach().clone().cpu().type(torch.int)
                            ]
                        )
                        .reshape(-1, 1)
                        .to(x.device),
                        torch.tensor(
                            self.l_scale[
                                x[..., -1].detach().clone().cpu().type(torch.int)
                            ]
                        )
                        .reshape(-1, 1)
                        .to(x.device),
                    )

                # Single size factor
                else:
                    l_loc, l_scale = self.l_loc * x.new_ones(
                        1
                    ), self.l_scale * x.new_ones(1)

                l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            # Part to modify if changing the decoder
            match self.reconstruction:
                case "ZINB":
                    rho_loc, rho_scale = self.rho_decoder(zw)
                    rho = pyro.sample(
                        "rho", dist.Normal(rho_loc, rho_scale).to_event(1)
                    )

                    # Append the batch
                    if self.batch_correction:
                        rho = torch.cat([rho, x[..., -1].view(-1, 1)], dim=-1)

                    gate_logits, mu = self.x_decoder(rho)
                    nb_logits = (l * mu + self.epsilon).log() - (
                        theta + self.epsilon
                    ).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits,
                        validate_args=False,
                    )

                case "Normal":
                    rho_loc, rho_scale = self.rho_decoder(zw)
                    rho = pyro.sample(
                        "rho", dist.Normal(rho_loc, rho_scale).to_event(1)
                    )

                    # Append the batch
                    if self.batch_correction:
                        rho = torch.cat([rho, x[..., -1].view(-1, 1)], dim=-1)

                    x_loc, x_scale = self.x_decoder(rho)
                    x_dist = dist.Normal(x_loc, x_scale)

                case "ZINB_LD":
                    # Append the batch
                    if self.batch_correction:
                        zw = torch.cat([zw, x[..., -1].view(-1, 1)], dim=-1)

                    gate_logits, mu = _split_in_half(self.x_decoder(zw))
                    mu = softmax(mu, dim=-1)
                    nb_logits = (l * mu + self.epsilon).log() - (
                        theta + self.epsilon
                    ).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits,
                        validate_args=False,
                    )

                case "Normal_LD":
                    # Append the batch
                    if self.batch_correction:
                        zw = torch.cat([zw, x[..., -1].view(-1, 1)], dim=-1)

                    _zw = zw.reshape(-1, zw.size(-1))
                    out = self.x_decoder(_zw)
                    out = out.reshape(zw.shape[:-1] + out.shape[-1:])

                    x_loc, x_scale = _split_in_half(out)
                    x_scale = softplus(x_scale)
                    x_dist = dist.Normal(x_loc, x_scale)

            with poutine.scale(None, self.recon_weight):
                # If batch corrected, we expect last index to be batch
                if self.batch_correction:
                    pyro.sample("x", x_dist.to_event(1), obs=x[..., :-1])
                else:
                    pyro.sample("x", x_dist.to_event(1), obs=x)

    # Guide
    def guide(self, x, y):
        """Approximate variational posterior for Patches.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y : :class:`~torch.Tensor`
            One-hot encoded and concatenated attribute labels.
        """
        pyro.module("patches", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # Variational for rho & l
            rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)

            if "ZINB" in self.reconstruction:
                pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            if self.reconstruction in ["ZINB_LD", "Normal_LD"]:
                rho = pyro.sample(
                    "rho",
                    dist.Normal(rho_loc, rho_scale).to_event(1),
                    infer={"is_auxiliary": True},
                )

            else:
                rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

            # Variational for w & z
            rho_y = _broadcast_inputs([rho, y])
            rho_y = torch.cat(rho_y, dim=-1)

            w_loc, w_scale = self.w_encoder(rho_y)
            z_loc, z_scale = self.z_encoder(rho)

            with poutine.scale(None, self.w_kl):
                pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

            with poutine.scale(None, self.z_kl):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # Classification for w (good) and z (bad)

            # Keep track over list
            classification_loss_z = 0
            attr_track = 0

            for i in pyro.plate("len_attrs", len(self.len_attrs)):
                next_track = attr_track + self.len_attrs[i]

                cur_func = getattr(self, f"classifier_z_y{i}")
                cur_logits = cur_func(z)
                cur_dist = dist.OneHotCategorical(logits=cur_logits)
                classification_loss_z += self.betas[i] * cur_dist.log_prob(
                    y[..., attr_track:next_track]
                )

                attr_track = next_track

            pyro.factor(
                "classification_loss", classification_loss_z, has_rsample=False
            )  # Want this maximized so positive sign in guide

            # TODO: rewrite with get weights to work for Normal_LD
            if (self.reconstruction in ["ZINB_LD"]) and self.sparsity:
                params = (
                    list(self.x_decoder.parameters())[0].T[self.latent_dim :].clone()
                )
                _, x_loc_params = params.reshape(params.shape[:-1] + (2, -1)).unbind(-2)
                pyro.factor(
                    "l1_loss",
                    x_loc_params.abs().sum().mul(self.sparsity_lambda),
                    has_rsample=False,
                )  # sparsity

    # Adverserial
    def adversarial(self, x, y):
        """Adversarial loss for Patches.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y : :class:`~torch.Tensor`
            One-hot encoded and concatenated attribute labels.
        """
        pyro.module("patches", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # Variational for rho & l
            rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)

            rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

            # Variational for w & z
            rho_y = _broadcast_inputs([rho, y])
            rho_y = torch.cat(rho_y, dim=-1)

            z_loc, z_scale = self.z_encoder(rho)

            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # Classification for w (good) and z (bad)

            # Keep track over list
            classification_loss_z = 0
            attr_track = 0

            for i in pyro.plate("len_atrrs", len(self.len_attrs)):
                next_track = attr_track + self.len_attrs[i]

                cur_func = getattr(self, f"classifier_z_y{i}")
                cur_logits = cur_func(z)
                cur_dist = dist.OneHotCategorical(logits=cur_logits)
                classification_loss_z += self.betas[i] * cur_dist.log_prob(
                    y[..., attr_track:next_track]
                )

                attr_track = next_track

            return -1.0 * classification_loss_z

    # Function to move points between conditions
    @torch.no_grad()
    def generate(self, x, y_source=None, y_target=None, w_custom=None):
        """Function used post-training for Patches to facilitate transfer between conditional labels.

        Parameters
        ----------
        x : :class:`~torch.Tensor`
            Input gene counts.

        y_source : :class:`~torch.Tensor`
            One-hot encoded and concatenated attribute labels for the input.

        y_target : :class:`~torch.Tensor`
            One-hot encoded and concatenated attribute labels for the target. Must be the same size in the first dimension as input.
        """
        pyro.module("patches", self)

        ## Encode
        # Variational for rho & l
        rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)

        rho_enc = pyro.sample("rho_enc", dist.Normal(rho_loc, rho_scale).to_event(1))
        l_enc = pyro.sample("l_enc", dist.LogNormal(l_loc, l_scale).to_event(1))

        # Variational for w & z
        ## TODO: Search the attribute space instead of picking a single sample

        # Keep tracked attributes in a list
        y_s = []
        attr_track = 0

        for i in range(len(self.len_attrs)):
            next_track = attr_track + self.len_attrs[i]
            y_s.append(y_target[..., attr_track:next_track])

            attr_track = next_track

        if w_custom is None:
            w_loc = torch.concat(
                [self._concat_lat_dims(y, self.w_locs, self.w_dim) for y in y_s], dim=-1
            )
            w_scale = torch.concat(
                [self._concat_lat_dims(y, self.w_scales, self.w_dim) for y in y_s],
                dim=-1,
            )

            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        else:
            w = w_custom

        z_loc, z_scale = self.z_encoder(rho_enc)
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        zw = torch.cat([z, w], dim=-1)

        match self.reconstruction:
            case "ZINB":
                rho_loc, rho_scale = self.rho_decoder(zw)
                rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

                # Append the batch
                if self.batch_correction:
                    rho = torch.cat([rho, x[..., -1].view(-1, 1)], dim=-1)

                gate_logits, mu = self.x_decoder(rho)

                nb_logits = (l_enc * mu + self.epsilon).log() - (
                    theta.to(mu.device) + self.epsilon
                ).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(
                    gate_logits=gate_logits,
                    total_count=theta,
                    logits=nb_logits,
                    validate_args=False,
                )

            case "Normal":
                rho_loc, rho_scale = self.rho_decoder(zw)
                rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

                # Append the batch
                if self.batch_correction:
                    rho = torch.cat([rho, x[..., -1].view(-1, 1)], dim=-1)

                x_loc, x_scale = self.x_decoder(rho)
                x_dist = dist.Normal(x_loc, x_scale)

            case "ZINB_LD":
                # Append the batch
                if self.batch_correction:
                    zw = torch.cat([zw, x[..., -1].view(-1, 1)], dim=-1)

                gate_logits, mu = _split_in_half(self.x_decoder(zw))
                mu = softmax(mu, dim=-1)
                nb_logits = (l_enc * mu + self.epsilon).log() - (
                    theta.to(mu.device) + self.epsilon
                ).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(
                    gate_logits=gate_logits,
                    total_count=theta,
                    logits=nb_logits,
                    validate_args=False,
                )

            case "Normal_LD":
                # Append the batch
                if self.batch_correction:
                    zw = torch.cat([zw, x[..., -1].view(-1, 1)], dim=-1)

                _zw = zw.reshape(-1, zw.size(-1))
                out = self.x_decoder(_zw)
                out = out.reshape(zw.shape[:-1] + out.shape[-1:])

                x_loc, x_scale = _split_in_half(out)
                x_scale = softplus(x_scale)
                x_dist = dist.Normal(x_loc, x_scale)

        # Observe the datapoint x using the observation distribution x_dist
        x_rec = pyro.sample("x", x_dist.to_event(1))

        return x_rec

    def get_weights(self):
        """Returns interpretable coefficients for latents.

        Refer to Notes for details.


        Returns
        -------
        loc, mu : :class:`~torch.Tensor`
            Mu of ZINB or Gaussian.

        scale, logits : :class:`~torch.Tensor`
            Either the variance of the Gaussian or ZI logits for ZINB.
        """
        assert self.reconstruction.endswith("LD")
        match self.reconstruction:
            case "ZINB_LD":
                if self.batch_correction:
                    logits, mu = _split_in_half(
                        list(self.x_decoder.parameters())[0].T[:-1].detach().cpu()
                    )
                else:
                    logits, mu = _split_in_half(
                        list(self.x_decoder.parameters())[0].T.detach().cpu()
                    )
                return mu, logits

            case "Normal_LD":
                if self.batch_correction:
                    loc, scale = _split_in_half(
                        list(self.x_decoder.parameters())[0].T[:-1].detach().cpu()
                    )
                else:
                    loc, scale = _split_in_half(
                        list(self.x_decoder.parameters())[0].T.detach().cpu()
                    )
                return loc, scale

    # Save self
    def save(self, path="patches_params"):
        """Saves model parameters to disk.

        Parameters
        ----------
        path : :class:`str`, default: "patches_params"
            Path to save model parameters.
        """
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")

    # Load
    def load(self, path="patches_params", map_location=None):
        """Loads model parameters from disk.

        Parameters
        ----------
        path : :class:`str`, default: "parches_params"
            Path to find model parameters. Should not include the extensions `_torch.pth` or `_pyro.pth` or any such variant.

        map_location : :class:`str`, optional
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


# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
