from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from anndata import AnnData

from ladder.module import GaussianDLVAE

from ._base import AdversarialInput, BaseDiscover, DecoderType

ReconstructionLikelihood = Literal["gaussian"]


class Discover(BaseDiscover):
    """Unit-scale Gaussian DLVAE with nonlinear or linear-mean decoders.

    Set ``decoder_type="linear"`` to train both reconstruction heads as affine
    maps of the latents and optional batch covariates. Continuous input features
    should be comparably scaled because the likelihood uses unit variance.
    """

    _registry_key = "_discover"
    _module_classes = {
        "gaussian": GaussianDLVAE,
    }
    _likelihood_param = "reconstruction_likelihood"

    def __init__(
        self,
        adata: AnnData,
        *,
        reconstruction_likelihood: ReconstructionLikelihood = "gaussian",
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
        decoder_type: DecoderType = "nonlinear",
    ):
        if w_dim is None:
            w_dim = latent_dim

        self._initialize(
            adata,
            likelihood=reconstruction_likelihood,
            init_params={
                "reconstruction_likelihood": reconstruction_likelihood,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "use_batch_norm": use_batch_norm,
                "dropout_rate": dropout_rate,
                "log_variational": log_variational,
                "latent_dim": latent_dim,
                "w_dim": w_dim,
                "recon_weight": recon_weight,
                "recon_weight_z": recon_weight_z,
                "z_kl_weight": z_kl_weight,
                "w_kl_weight": w_kl_weight,
                "adversarial_weight": adversarial_weight,
                "adversarial_input": adversarial_input,
                "classifier_layers": classifier_layers,
                "learnable_prior": learnable_prior,
                "decoder_type": decoder_type,
            },
        )

    def get_reconstruction(
        self,
        adata: AnnData | None = None,
        *,
        indices: Sequence[int] | Sequence[bool] | np.ndarray | None = None,
        z_only: bool = False,
        use_posterior_mean: bool = True,
        n_samples: int = 1,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return Gaussian reconstruction means for the selected observations.

        Posterior draws are averaged unless ``use_posterior_mean=True``. Set
        ``z_only=True`` to use the Z-only reconstruction head.
        """
        return self._decode_posterior(
            adata,
            indices=indices,
            z_only=z_only,
            use_posterior_mean=use_posterior_mean,
            n_samples=n_samples,
            batch_size=batch_size,
        )
