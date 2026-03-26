"""SIGReg: Sketched-Isotropic-Gaussian Regularizer.

Random unit-norm projections + Epps-Pulley normality test via trapezoid
quadrature. Forces the embedding distribution toward N(0, I) by the
Cramer-Wold theorem.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """SIGReg anti-collapse regularizer.

    Projects embeddings onto M random unit-norm directions, computes the
    Epps-Pulley univariate normality test statistic for each projection
    using trapezoid rule quadrature over t in [t_min, t_max], and averages.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_projections: int = 1024,
        num_knots: int = 17,
        t_min: float = 0.2,
        t_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.M = num_projections
        self.num_knots = num_knots
        self.t_min = t_min
        self.t_max = t_max

        # Pre-compute quadrature points and weights (not trainable)
        t_vals = torch.linspace(t_min, t_max, num_knots)
        self.register_buffer("t_vals", t_vals)

        # Gaussian weighting: w(t) = exp(-t^2/2)
        w = torch.exp(-t_vals ** 2 / 2)
        self.register_buffer("w", w)

    def _sample_directions(self, device: torch.device) -> torch.Tensor:
        """Sample M unit-norm random directions on S^{D-1}.

        Args:
            device: Target device.
        Returns:
            U: (D, M) matrix of unit-norm column vectors.
        """
        U = torch.randn(self.embed_dim, self.M, device=device)
        U = U / U.norm(dim=0, keepdim=True)
        return U

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss for a batch of embeddings (fully vectorized).

        Uses the identity: |phi_N(t)|^2 = (mean cos(t*h))^2 + (mean sin(t*h))^2
        to avoid O(N^2) pairwise computation.

        Args:
            Z: (B, D) embeddings for one timestep.
        Returns:
            Scalar SIGReg loss.
        """
        B, D = Z.shape

        # Sample random projection directions
        U = self._sample_directions(Z.device)  # (D, M)

        # Project embeddings onto all directions
        H = Z @ U  # (B, M)

        t = self.t_vals  # (K,)
        K = t.shape[0]
        M = self.M

        # Compute t * H for all knots, batch elements, projections
        # t_H: (K, B, M) = t[k] * H[b, m]
        t_H = t.view(K, 1, 1) * H.view(1, B, M)

        cos_tH = torch.cos(t_H)  # (K, B, M)
        sin_tH = torch.sin(t_H)  # (K, B, M)

        # |phi_N(t)|^2 = (mean_b cos(t*h))^2 + (mean_b sin(t*h))^2
        mean_cos = cos_tH.mean(dim=1)  # (K, M)
        mean_sin = sin_tH.mean(dim=1)  # (K, M)
        term1 = mean_cos ** 2 + mean_sin ** 2  # (K, M)

        # Cross term: 2 * mean_cos * phi_0(t) where phi_0(t) = exp(-t^2/2)
        exp_half = torch.exp(-t ** 2 / 2).view(K, 1)  # (K, 1)
        term2 = 2 * mean_cos * exp_half  # (K, M)

        # |phi_0(t)|^2 = exp(-t^2)
        term3 = torch.exp(-t ** 2).view(K, 1)  # (K, 1)

        # Integrand: w(t) * |phi_N(t) - phi_0(t)|^2
        integrand = self.w.view(K, 1) * (term1 - term2 + term3)  # (K, M)

        # Trapezoid integration over t for each projection
        dt = (self.t_max - self.t_min) / (self.num_knots - 1)
        result = torch.trapezoid(integrand, dx=dt, dim=0)  # (M,)

        # Average over M projections
        return result.mean()


def compute_sigreg_stepwise(
    embeddings: torch.Tensor, sigreg_module: SIGReg
) -> torch.Tensor:
    """Apply SIGReg step-wise across timesteps and average.

    SIGReg is applied independently to each timestep's batch of embeddings,
    then averaged over timesteps. This is the correct application per the paper.

    Args:
        embeddings: (B, T, D) trajectory of embeddings.
        sigreg_module: SIGReg instance.
    Returns:
        Scalar averaged SIGReg loss.
    """
    B, T, D = embeddings.shape
    total = torch.tensor(0.0, device=embeddings.device)
    for t in range(T):
        total = total + sigreg_module(embeddings[:, t, :])
    return total / T


if __name__ == "__main__":
    sigreg = SIGReg(embed_dim=192, num_projections=256, num_knots=17)

    # Test with standard Gaussian (should be near 0)
    Z_gaussian = torch.randn(128, 192)
    loss_gaussian = sigreg(Z_gaussian)
    print(f"SIGReg(Gaussian): {loss_gaussian.item():.6f} (should be near 0)")

    # Test with collapsed distribution (should be large)
    Z_collapsed = torch.zeros(128, 192) + 0.001 * torch.randn(128, 192)
    loss_collapsed = sigreg(Z_collapsed)
    print(f"SIGReg(collapsed): {loss_collapsed.item():.6f} (should be large)")

    # Test gradient flow
    Z_grad = torch.randn(32, 192, requires_grad=True)
    loss = sigreg(Z_grad)
    loss.backward()
    print(f"Gradient exists: {Z_grad.grad is not None}")
    print(f"Gradient norm: {Z_grad.grad.norm().item():.6f}")

    # Test step-wise
    emb = torch.randn(16, 4, 192)
    sw_loss = compute_sigreg_stepwise(emb, sigreg)
    print(f"Step-wise SIGReg: {sw_loss.item():.6f}")
