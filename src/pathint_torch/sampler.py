from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Optional, Tuple

import torch
import torch.distributions as dist
from torch import Tensor, nn
from torchsde import BrownianInterval, sdeint
from torchtyping import TensorType


class TrainingSDE(nn.Module):
    """
    SDE used for training. The drift and diffusion coefficients are functions of
    the augmented state variable :math:`(\\mathbf{x}_t, y_t)`, where :math:`\\mathbf{x}_t`
    is the state and :math:`y_t` is the cost.

    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model: nn.Module):
        """
        Args:
            model: control policy network taking :math:`t` and :math:`\\mathbf{x}_t`
            as arguments.
        """
        super().__init__()
        self.model = model

    def f_and_g_prod(
        self, t: TensorType[()], xy: TensorType["b", "s+1"], v: TensorType["b", "s"]
    ) -> Tuple[TensorType["b", "s+1"], TensorType["b", "s+1"]]:
        t = t[None].repeat(len(xy))
        x = xy[..., :-1]
        u = self.model(t, x)

        assert u.shape == x.shape
        assert v.shape == x.shape

        f = torch.cat((u, 0.5 * (u**2).sum(-1, keepdim=True)), -1)
        zeros = torch.zeros((len(t), 1), device=t.device, dtype=t.dtype)
        g_prod = torch.cat((v, zeros), -1)
        return f, g_prod


class SamplingSDE(nn.Module):
    """
    SDE used for sampling. The drift and diffusion coefficients are functions of
    the augmented state variable :math:`(\\mathbf{x}_t, y_t)`, where :math:`\\mathbf{x}_t`
    is the state and :math:`y_t` is the cost.

    """

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, model: nn.Module):
        """
        Args:
            model: control policy network taking :math:`t` and :math:`\\mathbf{x}_t`
            as arguments.
        """
        super().__init__()
        self.model = model

    def f_and_g_prod(
        self, t: TensorType[()], xy: TensorType["b", "s+1"], v: TensorType["b", "s"]
    ) -> Tuple[TensorType["b", "s+1"], TensorType["b", "s+1"]]:
        t = t[None].repeat(len(xy))
        x = xy[..., :-1]
        u = self.model(t, x)

        assert u.shape == x.shape
        assert v.shape == x.shape

        f = torch.cat((u, 0.5 * (u**2).sum(-1, keepdim=True)), -1)
        g_prod = torch.cat((v, (u * v).sum(-1, keepdim=True)), -1)
        return f, g_prod


@dataclass
class PathIntegralSampler:
    """
    Class defining loss and sampling functions for the path integral sampler.

    This approach consists of a training objective and sampling procedure for optimal
    control of the stochastic process

    .. math:: \\mathrm{d}\\mathbf{x}_t = \\mathbf{u}_t \\mathrm{d}t + \\mathrm{d}\\mathbf{w}_t ,

    where :math:`\\mathbf{w}_t` is a Wiener process. A network trained to find
    the control policy :math:`\\mathbf{u}_t(t, \\mathbf{x})` such that the loss
    function is minimized causes the above process to yield samples at time :math:`T`
    with the prespecified distribution :math:`\\mu(\\cdot)`. (Distributions and
    quantities at time :math:`t=T` are often referred to as "terminal".) The procedure
    also yields importance sampling weights :math:`w`.

    The undocumented attributes are keyword arguments passed to `torchsde.sdeint <https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md>`_.

    Notes:
        As explained in the paper, the control policy network is trained by constructing
        an SDE augmented by the trajectory's cost. This implementation uses a similar
        trick to simultaneously sample and compute importance sampling weights
        using any SDE solver.

    """

    get_log_mu: Callable[[Tensor], Tensor]
    """:math:`\\log \\mu(x)`, the log of the (unnormalized) terminal density to
    be sampled from.
    """
    x_size: int
    """size of :math:`x` vector."""
    T: float
    """duration of diffusion."""
    dt: float
    """initial timestep size for solver."""
    method: str = "euler"
    """SDE solver."""
    adaptive: bool = False
    rtol: float = 1e-5
    atol: float = 1e-4
    dt_min: float = 1e-5
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    y0: TensorType["s"] = field(init=False)
    """point at which diffusion begins (the origin)."""
    mu_0: dist.Distribution = field(init=False)
    """terminal distribution of uncontrolled process."""

    def __post_init__(self):
        self.y0 = torch.zeros(self.x_size + 1, device=self.device, dtype=self.dtype)
        self.mu_0 = dist.Normal(loc=self.y0[:-1], scale=sqrt(self.T))

    def _sample_xs_cost(
        self,
        sde: nn.Module,
        ts: TensorType["t"],
        batch_size: int,
        entropy: Optional[int] = None,
    ) -> Tuple[TensorType["t", "b", "s"], TensorType["b"]]:
        """
        Helper to generate sample paths with their costs.
        """
        # Batch the initial conditions
        y0 = self.y0[None, :].repeat((batch_size, 1))
        bm = BrownianInterval(
            0.0,
            self.T,
            (batch_size, self.x_size),
            self.dtype,
            self.device,
            entropy,
            dt=self.dt,
        )
        path = sdeint(
            sde,
            y0,
            ts,
            bm=bm,
            method=self.method,
            dt=self.dt,
            adaptive=self.adaptive,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=self.dt_min,
        )
        assert isinstance(path, Tensor)  # to shut typing up
        # Get path
        xs = path[:, :, :-1]
        # Get cost
        ys = path[-1, :, -1]
        # Add terminal cost
        Psi_T = self.mu_0.log_prob(xs[-1]).sum(-1) - self.get_log_mu(xs[-1])
        ys += Psi_T
        return xs, ys

    def sample_loss(
        self, model: nn.Module, batch_size: int, entropy: Optional[int] = None
    ) -> TensorType["b"]:
        """
        Gets loss for a single trajectory.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            batch_size: batch size.
            n_intermediate: number of intermediate timesteps at which to save the
                trajectory.
            entropy: seed for Brownian motion.

        Returns:
            cost: approximation to :math:`\\int_{t_0}^{t_1} \\mathrm{d}t \\frac{1}{2} \\mathbf{u}_t(t, \\mathbf{x}_t ; \\theta) + \\Psi(\\mathbf{x}_T)`,
                where the second term is the terminal cost specified by the training
                procedure.
        """
        ts = torch.tensor((0.0, self.T), device=self.device, dtype=self.dtype)
        sde = TrainingSDE(model)
        return self._sample_xs_cost(sde, ts, batch_size, entropy)[1]

    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        n_intermediate: int = 0,
        entropy: Optional[int] = None,
    ) -> Tuple[TensorType["t", "b", "s"], TensorType["b"]]:
        """
        Generates a sample. To generate multiple samples, `vmap` over `key`.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            batch_size: batch size.
            n_intermediate: number of intermediate timesteps at which to save the
                trajectory.
            entropy: seed for Brownian motion.

        Returns:
            xs: sample paths at :math:`t = 0`, :math:`T` and `n_intermediate` times in between.
            log_w: corresponding log importance sampling weights.
        """
        ts = torch.linspace(
            0.0, self.T, 2 + n_intermediate, device=self.device, dtype=self.dtype
        )
        sde = TrainingSDE(model)
        xs, ys = self._sample_xs_cost(sde, ts, batch_size, entropy)
        # Convert cost into log weight
        log_w = -ys
        return xs, log_w
