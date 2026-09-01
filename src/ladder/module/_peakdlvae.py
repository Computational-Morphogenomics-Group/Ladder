from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from torch import Tensor, nn
from torch.distributions import Distribution

from ._base import AdversarialInput, BaseDLVAE, DecoderType
from ._constants import EPS


class PoissonPEAKDLVAE(BaseDLVAE):
    """snATAC-seq DLVAE with a Poisson peak-count likelihood."""

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
        super().__init__(
            in_dim=in_dim,
            label_dims=label_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            log_variational=log_variational,
            latent_dim=latent_dim,
            w_dim=w_dim,
            recon_weight=recon_weight,
            recon_weight_z=recon_weight_z,
            z_kl_weight=z_kl_weight,
            w_kl_weight=w_kl_weight,
            adversarial_weight=adversarial_weight,
            adversarial_input=adversarial_input,
            classifier_layers=classifier_layers,
            learnable_prior=learnable_prior,
            covariate_dim=covariate_dim,
            decoder_type=decoder_type,
        )
        self.decoder.layers[-1].bias = None
        self.decoder_z.layers[-1].bias = None
        self.region_logits = nn.Parameter(torch.zeros(self.in_dim))

    def decode(
        self,
        z: Tensor,
        w: Tensor | None = None,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Decode normalized accessibility from Z alone or from W and Z."""
        if w is None:
            return self._normalized_accessibility(self.decoder_z, z, batch_covariates)
        w_z = torch.cat((w, z), dim=-1)
        return self._normalized_accessibility(self.decoder, w_z, batch_covariates)

    def _observed_library_size(self, x: Tensor) -> Tensor:
        return x.sum(-1, keepdim=True).clamp_min(EPS)

    def _region_logits_like(self, reference: Tensor) -> Tensor:
        return self.region_logits.unsqueeze(0).type_as(reference)

    def _poisson_distribution(
        self,
        decoder: nn.Module,
        latent: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> tuple[Distribution, Tensor]:
        decoder_logits = decoder(self._decoder_input(latent, batch_covariates))
        normalized_accessibility = torch.softmax(decoder_logits, dim=-1)
        region_logits = self._region_logits_like(decoder_logits)
        region_adjusted_accessibility = torch.softmax(
            decoder_logits + region_logits, dim=-1
        )
        library_size = self._observed_library_size(x).type_as(
            region_adjusted_accessibility
        )
        rate = library_size * region_adjusted_accessibility
        return dist.Poisson(rate, validate_args=False), normalized_accessibility

    def _normalized_accessibility(
        self,
        decoder: nn.Module,
        latent: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        decoder_logits = decoder(self._decoder_input(latent, batch_covariates))
        return torch.softmax(decoder_logits, dim=-1)

    def _reconstruct(
        self,
        w_z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        x_dist, normalized_accessibility = self._poisson_distribution(
            self.decoder,
            w_z,
            x,
            batch_covariates,
        )
        with poutine.scale(scale=self.recon_weight):
            pyro.sample("rec_w", x_dist.to_event(1), obs=x)
        return normalized_accessibility

    def _reconstruct_z(
        self,
        z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        x_dist, normalized_accessibility = self._poisson_distribution(
            self.decoder_z,
            z,
            x,
            batch_covariates,
        )
        with poutine.scale(scale=self.recon_weight_z):
            pyro.sample("rec_z", x_dist.to_event(1), obs=x)
        return normalized_accessibility

    def _reconstruction_distribution(
        self,
        decoder: nn.Module,
        latent: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> tuple[Distribution, Tensor]:
        x_dist, _ = self._poisson_distribution(
            decoder,
            latent,
            x,
            batch_covariates,
        )
        return x_dist, x
