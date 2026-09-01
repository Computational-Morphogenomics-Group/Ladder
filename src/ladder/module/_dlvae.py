from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from torch import Tensor, nn
from torch.distributions import Distribution

from ._base import BaseDLVAE


class GaussianDLVAE(BaseDLVAE):
    """DLVAE with a unit-scale Gaussian reconstruction likelihood."""

    def decode(
        self,
        z: Tensor,
        w: Tensor | None = None,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        """Decode the reconstruction mean from Z alone or from W and Z."""
        if w is None:
            return self._decoder_loc(self.decoder_z, z, batch_covariates)
        w_z = torch.cat((w, z), dim=-1)
        return self._decoder_loc(self.decoder, w_z, batch_covariates)

    def _decoder_loc(
        self,
        decoder: nn.Module,
        latent: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        return decoder(self._decoder_input(latent, batch_covariates))

    def _reconstruct(
        self,
        w_z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        x_dist, obs = self._reconstruction_distribution(
            self.decoder,
            w_z,
            x,
            batch_covariates,
        )
        loc = x_dist.mean

        with poutine.scale(scale=self.recon_weight):
            pyro.sample("rec_w", x_dist.to_event(1), obs=obs)

        return loc

    def _reconstruct_z(
        self,
        z: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> Tensor:
        x_dist, obs = self._reconstruction_distribution(
            self.decoder_z,
            z,
            x,
            batch_covariates,
        )
        loc = x_dist.mean

        with poutine.scale(scale=self.recon_weight_z):
            pyro.sample("rec_z", x_dist.to_event(1), obs=obs)

        return loc

    def _reconstruction_distribution(
        self,
        decoder: nn.Module,
        latent: Tensor,
        x: Tensor,
        batch_covariates: Tensor | None = None,
    ) -> tuple[Distribution, Tensor]:
        loc = self._decoder_loc(decoder, latent, batch_covariates)
        scale = torch.ones_like(loc)
        x_dist = dist.Normal(loc, scale, validate_args=False)
        return x_dist, x
