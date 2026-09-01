from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from anndata import AnnData

from ladder.module import PoissonPEAKDLVAE

from ._base import AdversarialInput, BaseDiscover, DecoderType

PeakLikelihood = Literal["poisson"]


class PeakDiscover(BaseDiscover):
    """Poisson ATAC DLVAE with nonlinear or linear-logit decoders.

    Set ``decoder_type="linear"`` to learn linear logits in both reconstruction
    heads. The Poisson rate uses shared region intercepts and each cell's
    observed fragment total. Input data should be raw peak counts, without
    binarization.
    """

    _registry_key = "_peakdiscover"
    _module_classes = {
        "poisson": PoissonPEAKDLVAE,
    }
    _likelihood_param = "peak_likelihood"

    def __init__(
        self,
        adata: AnnData,
        *,
        peak_likelihood: PeakLikelihood = "poisson",
        hidden_dim: int | None = None,
        num_layers: int = 2,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        log_variational: bool = False,
        latent_dim: int | None = None,
        w_dim: int | None = None,
        recon_weight: float = 0.8,
        recon_weight_z: float = 0.2,
        z_kl_weight: float = 1e-2,
        w_kl_weight: float = 1e-2,
        adversarial_weight: float = 250.0,
        adversarial_input: AdversarialInput = "reconstruction",
        classifier_layers: int = 1,
        learnable_prior: bool = False,
        decoder_type: DecoderType = "nonlinear",
    ):
        if hidden_dim is None:
            hidden_dim = int(np.sqrt(adata.n_vars))
        if latent_dim is None:
            latent_dim = int(np.sqrt(hidden_dim))
        if w_dim is None:
            w_dim = latent_dim

        self._initialize(
            adata,
            likelihood=peak_likelihood,
            init_params={
                "peak_likelihood": peak_likelihood,
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

    def get_normalized_accessibility(
        self,
        adata: AnnData | None = None,
        *,
        indices: Sequence[int] | Sequence[bool] | np.ndarray | None = None,
        z_only: bool = False,
        use_posterior_mean: bool = False,
        n_samples: int = 1,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return latent-driven relative accessibility for selected observations.

        The output is the softmax of the decoder logits, excluding the learned
        region intercepts and observed fragment totals used in the Poisson rate.
        Posterior draws are averaged unless ``use_posterior_mean=True``.
        """
        return self._decode_posterior(
            adata,
            indices=indices,
            z_only=z_only,
            use_posterior_mean=use_posterior_mean,
            n_samples=n_samples,
            batch_size=batch_size,
        )
