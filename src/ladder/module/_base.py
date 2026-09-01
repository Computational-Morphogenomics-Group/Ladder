from __future__ import annotations

from typing import Literal

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro import poutine
from torch import Tensor, nn
from torch.distributions import Distribution

from ._constants import EPS
from ._layers import MLP, GaussianMLP

AdversarialInput = Literal["reconstruction", "z"]
DecoderType = Literal["nonlinear", "linear"]


class BaseDLVAE(nn.Module):
    """Shared latent-variable and adversarial mechanics for DLVAE modules.

    The guide parameterizes q(z | x) and q(w | x, y). The model reconstructs
    x with full W-and-Z and Z-only decoder heads. ``decoder_type="linear"``
    removes hidden layers from both decoders only.
    """

    def __init__(
        self,
        in_dim: int,
        label_dims: list[int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        log_variational: bool = False,
        latent_dim: int = 10,
        w_dim: int | None = None,
        recon_weight: float = 1.0,
        recon_weight_z: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        adversarial_input: AdversarialInput = "reconstruction",
        classifier_layers: int = 1,
        learnable_prior: bool = False,
        covariate_dim: int = 0,
        decoder_type: DecoderType = "nonlinear",
    ):
        super().__init__()

        if w_dim is None:
            w_dim = latent_dim
        if not learnable_prior and w_dim != latent_dim:
            raise ValueError("w_dim must equal latent_dim when learnable_prior=False.")

        self.in_dim = in_dim
        self.label_dims = label_dims
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.log_variational = log_variational
        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.recon_weight = recon_weight
        self.recon_weight_z = recon_weight_z
        self.z_kl_weight = z_kl_weight
        self.w_kl_weight = w_kl_weight
        self.adversarial_weight = adversarial_weight
        self.adversarial_input = adversarial_input
        self.classifier_layers = classifier_layers
        self.learnable_prior = learnable_prior
        self.covariate_dim = covariate_dim
        self.decoder_type = decoder_type

        hidden_dims = [hidden_dim] * num_layers
        decoder_hidden_dims = [] if decoder_type == "linear" else hidden_dims
        label_dim = sum(label_dims)

        self.encoder = GaussianMLP(
            in_dim,
            hidden_dims,
            latent_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )
        self.encoder_w = GaussianMLP(
            in_dim + label_dim,
            hidden_dims,
            w_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )
        self.decoder = MLP(
            w_dim + latent_dim + covariate_dim,
            decoder_hidden_dims,
            in_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )
        self.decoder_z = MLP(
            latent_dim + covariate_dim,
            decoder_hidden_dims,
            in_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        if learnable_prior:
            self.prior_w = MLP(
                latent_dim + label_dim,
                [hidden_dim],
                w_dim,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
            )

        classifier_input_dim = latent_dim if adversarial_input == "z" else in_dim
        classifier_dims = [dim if dim != 1 else 2 for dim in label_dims]
        self.classifiers = nn.ModuleList(
            [
                MLP(
                    classifier_input_dim,
                    [hidden_dim] * classifier_layers,
                    dim,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                )
                for dim in classifier_dims
            ]
        )

    def model(
        self,
        x: Tensor,
        y: Tensor,
        batch_covariates: Tensor | None = None,
    ):
        """Define the Pyro model and model-side adversarial objective."""
        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=1.0 / x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.latent_dim))
            z_scale = x.new_ones((x.shape[0], self.latent_dim))

            with poutine.scale(scale=self.z_kl_weight):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            w_loc, w_scale = self._w_prior_parameters(y, z)

            with poutine.scale(scale=self.w_kl_weight):
                w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

            self._reconstruct(torch.cat((w, z), dim=-1), x, batch_covariates)
            rec_z = self._reconstruct_z(z, x, batch_covariates)

            if self.adversarial_weight != 0:
                adversarial_features = self._adversarial_features(
                    z,
                    batch_covariates,
                    decoded_z=rec_z,
                )
                with poutine.scale(scale=self.adversarial_weight):
                    pyro.factor(
                        "adversarial_loss",
                        -self._negative_entropy(adversarial_features),
                    )

    def guide(
        self,
        x: Tensor,
        y: Tensor,
        batch_covariates: Tensor | None = None,
    ):
        """Define q(z | x) and q(w | x, y) for Pyro ELBO estimation."""
        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=1.0 / x.shape[0]):
            encoder_x = self._encoder_input(x)
            z_loc, z_scale = self.encoder(encoder_x)

            with poutine.scale(scale=self.z_kl_weight):
                pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            w_loc, w_scale = self.encoder_w(torch.cat((encoder_x, y), dim=-1))

            with poutine.scale(scale=self.w_kl_weight):
                pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

    def classifier_loss(
        self,
        x: Tensor,
        y: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Return classifier negative log likelihood for a posterior sample of Z."""
        z_loc, z_scale = self.encoder(self._encoder_input(x))
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
        adversarial_features = self._adversarial_features(
            z,
            batch_covariates,
        )
        return -self._classifier_log_prob(adversarial_features, y)

    def inference(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Return the parameters of q(z | x) and q(w | x, y)."""
        encoder_x = self._encoder_input(x)
        z_loc, z_scale = self.encoder(encoder_x)
        w_loc, w_scale = self.encoder_w(torch.cat((encoder_x, y), dim=-1))

        return {
            "z_loc": z_loc,
            "z_scale": z_scale,
            "w_loc": w_loc,
            "w_scale": w_scale,
        }

    def decode(
        self,
        z: Tensor,
        w: Tensor | None = None,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Decode model-specific feature values from Z or from W and Z."""
        raise NotImplementedError

    def reconstruction_mean(
        self,
        z: Tensor,
        x: Tensor,
        w: Tensor | None = None,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Return the observation-likelihood mean using nuisance information from X."""
        if w is None:
            decoder = self.decoder_z
            latent = z
        else:
            decoder = self.decoder
            latent = torch.cat((w, z), dim=-1)

        reconstruction_dist, _ = self._reconstruction_distribution(
            decoder,
            latent,
            x,
            batch_covariates,
        )
        return reconstruction_dist.mean

    def loss_components(
        self,
        x: Tensor,
        y: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return batch-mean loss diagnostics evaluated at posterior means."""
        inference_outputs = self.inference(x, y)

        z_loc = inference_outputs["z_loc"]
        z_scale = inference_outputs["z_scale"]
        w_loc = inference_outputs["w_loc"]
        w_scale = inference_outputs["w_scale"]
        w_prior_loc, w_prior_scale = self._w_prior_parameters(y, z_loc)

        w_z = torch.cat((w_loc, z_loc), dim=-1)
        rec_w_dist, rec_w_obs = self._reconstruction_distribution(
            self.decoder,
            w_z,
            x,
            batch_covariates,
        )
        rec_z_dist, rec_z_obs = self._reconstruction_distribution(
            self.decoder_z,
            z_loc,
            x,
            batch_covariates,
        )
        reconstruction_w_nll = -rec_w_dist.log_prob(rec_w_obs).sum(dim=-1).mean()
        reconstruction_z_nll = -rec_z_dist.log_prob(rec_z_obs).sum(dim=-1).mean()

        z_kl = self._normal_kl(
            z_loc, z_scale, torch.zeros_like(z_loc), torch.ones_like(z_scale)
        )
        w_kl = self._normal_kl(w_loc, w_scale, w_prior_loc, w_prior_scale)
        z_kl = z_kl.sum(dim=-1).mean()
        w_kl = w_kl.sum(dim=-1).mean()

        adversarial_features = self._adversarial_features(
            z_loc,
            batch_covariates,
        )
        adversarial_entropy = -self._negative_entropy(adversarial_features)

        weighted_reconstruction_w_nll = self.recon_weight * reconstruction_w_nll
        weighted_reconstruction_z_nll = self.recon_weight_z * reconstruction_z_nll
        weighted_z_kl = self.z_kl_weight * z_kl
        weighted_w_kl = self.w_kl_weight * w_kl
        weighted_adversarial_loss = -self.adversarial_weight * adversarial_entropy
        weighted_decomposed_loss = (
            weighted_reconstruction_w_nll
            + weighted_reconstruction_z_nll
            + weighted_z_kl
            + weighted_w_kl
            + weighted_adversarial_loss
        )

        return {
            "reconstruction_w_nll": reconstruction_w_nll,
            "reconstruction_z_nll": reconstruction_z_nll,
            "z_kl": z_kl,
            "w_kl": w_kl,
            "adversarial_entropy": adversarial_entropy,
            "weighted_reconstruction_w_nll": weighted_reconstruction_w_nll,
            "weighted_reconstruction_z_nll": weighted_reconstruction_z_nll,
            "weighted_z_kl": weighted_z_kl,
            "weighted_w_kl": weighted_w_kl,
            "weighted_adversarial_loss": weighted_adversarial_loss,
            "weighted_decomposed_loss": weighted_decomposed_loss,
        }

    def _encoder_input(self, x: Tensor) -> Tensor:
        return torch.log1p(x) if self.log_variational else x

    def _decoder_input(
        self,
        latent: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        if self.covariate_dim == 0:
            return latent
        if batch_covariates is None:
            raise ValueError(
                "This module was initialized with covariates but no covariate tensor was provided."
            )
        return torch.cat((latent, batch_covariates), dim=-1)

    def _w_prior_parameters(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        if self.learnable_prior:
            w_loc = self.prior_w(torch.cat((z, y), dim=-1))
            w_scale = z.new_ones((y.shape[0], self.w_dim))
            return w_loc, w_scale

        _, group, counts = y.int().unique(
            dim=0, return_inverse=True, return_counts=True
        )
        group_sums = z.new_zeros((counts.numel(), self.w_dim))
        group_sums.index_add_(0, group, z)
        group_means = group_sums / counts[:, None]

        w_loc = group_means[group]
        w_scale = torch.ones_like(w_loc)
        return w_loc, w_scale

    def _classifier_log_prob(self, features: Tensor, y: Tensor) -> Tensor:
        log_prob = features.new_zeros((features.shape[0],))
        label_start = 0

        for classifier, label_dim in zip(
            self.classifiers, self.label_dims, strict=True
        ):
            label = y[..., label_start : label_start + label_dim]

            if label_dim == 1:
                label = F.one_hot(label.long().squeeze(-1), num_classes=2).to(
                    dtype=features.dtype
                )

            log_prob += dist.OneHotCategorical(
                logits=classifier(features),
            ).log_prob(label)

            label_start += label_dim

        return log_prob.mean()

    def _negative_entropy(self, features: Tensor) -> Tensor:
        negative_entropy = features.new_zeros(())

        for classifier in self.classifiers:
            log_p = F.log_softmax(classifier(features), dim=-1)
            negative_entropy += (log_p.exp() * log_p).sum(dim=-1).mean()

        return negative_entropy

    def _adversarial_features(
        self,
        z: Tensor,
        batch_covariates: Tensor | None = None,
        *,
        decoded_z: Tensor | None = None,
    ) -> Tensor:
        if self.adversarial_input == "z":
            return z
        if decoded_z is not None:
            return decoded_z
        return self.decode(z, batch_covariates=batch_covariates)

    @staticmethod
    def _normal_kl(
        loc: Tensor,
        scale: Tensor,
        prior_loc: Tensor,
        prior_scale: Tensor,
    ) -> Tensor:
        scale = scale.clamp_min(EPS)
        prior_scale = prior_scale.clamp_min(EPS)
        return (
            torch.log(prior_scale / scale)
            + (scale.square() + (loc - prior_loc).square())
            / (2.0 * prior_scale.square())
            - 0.5
        )

    def _reconstruct(
        self,
        w_z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Register the W-and-Z reconstruction term and return decoded values."""
        raise NotImplementedError

    def _reconstruct_z(
        self,
        z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Register the Z-only reconstruction term and return decoded values."""
        raise NotImplementedError

    def _reconstruction_distribution(
        self,
        decoder: nn.Module,
        latent: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> tuple[Distribution, Tensor]:
        """Return the observation distribution and values scored by its log probability."""
        raise NotImplementedError
