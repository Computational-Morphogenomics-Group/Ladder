from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from anndata import AnnData

from ladder.module import NBSCDLVAE

from ._base import AdversarialInput, BaseDiscover, DecoderType

GeneLikelihood = Literal["nb"]


class SCDiscover(BaseDiscover):
    """Negative binomial RNA DLVAE with nonlinear or linear-logit decoders.

    Set ``decoder_type="linear"`` to learn linear logits in both reconstruction
    heads. The likelihood uses each cell's observed library size and gene-specific
    inverse dispersion. Input data should be raw gene counts.
    """

    _registry_key = "_scdiscover"
    _module_classes = {
        "nb": NBSCDLVAE,
    }
    _likelihood_param = "gene_likelihood"

    def __init__(
        self,
        adata: AnnData,
        *,
        gene_likelihood: GeneLikelihood = "nb",
        hidden_dim: int = 128,
        num_layers: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        log_variational: bool = False,
        latent_dim: int = 10,
        w_dim: int | None = None,
        recon_weight: float = 0.8,
        recon_weight_z: float = 0.2,
        z_kl_weight: float = 1e-2,
        w_kl_weight: float = 1e-2,
        adversarial_weight: float = 100.0,
        adversarial_input: AdversarialInput = "reconstruction",
        classifier_layers: int = 1,
        learnable_prior: bool = False,
        decoder_type: DecoderType = "nonlinear",
    ):
        if w_dim is None:
            w_dim = latent_dim

        self._initialize(
            adata,
            likelihood=gene_likelihood,
            init_params={
                "gene_likelihood": gene_likelihood,
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

    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        *,
        indices: Sequence[int] | Sequence[bool] | np.ndarray | None = None,
        z_only: bool = False,
        use_posterior_mean: bool = False,
        n_samples: int = 1,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return decoder-normalized expression for selected observations.

        The output is the softmax of the decoder logits. Observed library sizes
        and gene-specific inverse dispersion are used only to construct the
        negative binomial likelihood. Posterior draws are averaged unless
        ``use_posterior_mean=True``.
        """
        return self._decode_posterior(
            adata,
            indices=indices,
            z_only=z_only,
            use_posterior_mean=use_posterior_mean,
            n_samples=n_samples,
            batch_size=batch_size,
        )
