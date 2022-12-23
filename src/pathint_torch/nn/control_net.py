from typing import Callable, List, Optional

import torch
from torch import nn
from torchtyping import TensorType
from torchvision.ops import MLP

from .positional_encoding import PositionalEncoding


class ControlNet(nn.Module):
    """
    Affine transformation of score parametrized by two neural networks, using the
    architecture close to the one from the `path integral sampler paper <https://arxiv.org/abs/2111.15141>`_.
    This is of the form :math:`a_\\theta(t, x) + b_\\theta(t, x) \\nabla \\log \\mu(x)`,
    with the output initialized to zero using an overall learn multiplicative factor.
    """

    def __init__(
        self,
        x_size: int,
        get_score_mu: Callable[[TensorType["b", "x"]], TensorType["b", "x"]],
        T: float,
        L_max: int,
        emb_dim: int,
        emb_hidden_widths: List[int],
        hidden_widths: List[int],
        norm_layer: Optional[nn.Module] = None,
        activation_layer=nn.LeakyReLU,
        scalar_coeff_net: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            x_size: size of :math:`x` vector.
            get_score_mu: score of the target density, :math:`\\nabla \\log \\mu(x)`.
            T: duration of diffusion.
            L_max: :math:`L` parameter for positional encoding of :math:`t`.
            emb_dim: dimension for embedding of :math:`t` and :math:`x`.
            scalar_coeff_net: if `True`, :math:`b_\\theta(t, x)` will output a scalar
                instead of a vector to be multiplied elementwise with :math:`\\nabla \\log \\mu(x)`.
        """
        super().__init__()
        self.get_score_mu = get_score_mu
        self.T = T

        # Embedding layers
        self.t_emb_net = nn.Sequential(
            PositionalEncoding(L_max),
            MLP(
                2 * L_max,
                emb_hidden_widths + [emb_dim],
                norm_layer,
                activation_layer,
                dropout=dropout,
            ),
        )
        self.x_emb_net = MLP(
            x_size,
            emb_hidden_widths + [emb_dim],
            norm_layer,
            activation_layer,
            dropout=dropout,
        )

        # Transformation layers
        self.const_net = MLP(
            2 * emb_dim,
            hidden_widths + [x_size],
            norm_layer,
            activation_layer,
            dropout=dropout,
        )
        self.coeff_net = MLP(
            2 * emb_dim,
            hidden_widths + [1 if scalar_coeff_net else x_size],
            norm_layer,
            activation_layer,
            dropout=dropout,
        )

        # Initialize final linear layers to zero
        for final_linear in [self.const_net[-2], self.coeff_net[-2]]:
            assert isinstance(final_linear, nn.Linear)
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward(
        self, t: TensorType["b"], x: TensorType["b", "x"]
    ) -> TensorType["b", "x"]:
        """
        Runs the network.
        """
        t_norm = t / self.T - 0.5
        t_emb = self.t_emb_net(t_norm)
        x_emb = self.x_emb_net(x)
        tx_emb = torch.cat((t_emb, x_emb), -1)

        const = self.const_net(tx_emb)
        coeff = self.coeff_net(tx_emb)
        score = self.get_score_mu(x)

        return const + coeff * score
