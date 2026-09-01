import torch.nn.functional as F
from torch import Tensor, nn

from ._constants import EPS


class MLP(nn.Module):
    """MLP with optional BatchNorm and dropout in hidden layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []

        for hidden_in, hidden_out in zip(dims[:-2], dims[1:-1], strict=True):
            layers.append(nn.Linear(hidden_in, hidden_out))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_out))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class GaussianMLP(MLP):
    """MLP returning the location and positive scale of a diagonal Gaussian."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=2 * out_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        flat_x = x.reshape(-1, x.shape[-1])
        parameters = super().forward(flat_x)
        parameters = parameters.reshape(x.shape[:-1] + parameters.shape[-1:])

        loc, unconstrained_scale = parameters.chunk(2, dim=-1)
        scale = F.softplus(unconstrained_scale) + EPS
        return loc, scale
