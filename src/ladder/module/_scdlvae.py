from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro import poutine
from torch import Tensor, nn
from torch.distributions import Distribution

from ._base import AdversarialInput, BaseDLVAE, DecoderType
from ._constants import EPS


class NBSCDLVAE(BaseDLVAE):
    """scRNA-seq DLVAE with a negative binomial gene likelihood."""

    _INITIAL_INVERSE_DISPERSION = 10.0

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
        initial_inverse_dispersion = torch.full(
            (self.in_dim,),
            self._INITIAL_INVERSE_DISPERSION,
        )
        self.inverse_dispersion = nn.Parameter(
            torch.log(torch.expm1(initial_inverse_dispersion))
        )

    def decode(
        self,
        z: Tensor,
        w: Tensor | None = None,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Decode normalized expression from Z alone or from W and Z."""
        if w is None:
            return self._normalized_expression(self.decoder_z, z, batch_covariates)
        w_z = torch.cat((w, z), dim=-1)
        return self._normalized_expression(self.decoder, w_z, batch_covariates)

    def _observed_library_size(self, x: Tensor) -> Tensor:
        return x.sum(dim=-1, keepdim=True).clamp_min(EPS)

    def _inverse_dispersion(self) -> Tensor:
        return F.softplus(self.inverse_dispersion) + EPS

    def _normalized_expression(
        self,
        decoder: nn.Module,
        latent: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        logits = decoder(self._decoder_input(latent, batch_covariates))
        return torch.softmax(logits, dim=-1)

    def _negative_binomial_distribution(
        self,
        decoder: nn.Module,
        latent: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> tuple[Distribution, Tensor]:
        decoder_logits = decoder(self._decoder_input(latent, batch_covariates))
        log_normalized_expression = F.log_softmax(decoder_logits, dim=-1)
        normalized_expression = log_normalized_expression.exp()
        library_size = self._observed_library_size(x).type_as(decoder_logits)
        inverse_dispersion = self._inverse_dispersion().type_as(decoder_logits)
        nb_logits = (
            library_size.log() + log_normalized_expression - inverse_dispersion.log()
        )
        reconstruction_dist = dist.NegativeBinomial(
            total_count=inverse_dispersion,
            logits=nb_logits,
            validate_args=False,
        )
        return reconstruction_dist, normalized_expression

    def _reconstruct(
        self,
        w_z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        x_dist, normalized_expression = self._negative_binomial_distribution(
            self.decoder,
            w_z,
            x,
            batch_covariates,
        )
        with poutine.scale(scale=self.recon_weight):
            pyro.sample("rec_w", x_dist.to_event(1), obs=x)
        return normalized_expression

    def _reconstruct_z(
        self,
        z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        x_dist, normalized_expression = self._negative_binomial_distribution(
            self.decoder_z,
            z,
            x,
            batch_covariates,
        )
        with poutine.scale(scale=self.recon_weight_z):
            pyro.sample("rec_z", x_dist.to_event(1), obs=x)
        return normalized_expression

    def _reconstruction_distribution(
        self,
        decoder: nn.Module,
        latent: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> tuple[Distribution, Tensor]:
        x_dist, _ = self._negative_binomial_distribution(
            decoder,
            latent,
            x,
            batch_covariates,
        )
        return x_dist, x
