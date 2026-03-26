"""Adaptive Layer Normalization with zero initialization for action conditioning."""

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with zero-initialized conditioning.

    At initialization, the MLP that produces scale and shift has zero weights
    and zero bias, so scale=0 and shift=0. Combined with a residual connection,
    the block initially acts as an identity. The model gradually learns to
    incorporate action conditioning during training.

    Follows the DiT paper convention (Peebles & Xie, 2023).
    """

    def __init__(self, embed_dim: int, cond_dim: int) -> None:
        """Initialize AdaLN.

        Args:
            embed_dim: Dimension of input features (192).
            cond_dim: Dimension of conditioning signal (action embedding).
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        # Project conditioning to scale and shift
        self.cond_proj = nn.Linear(cond_dim, 2 * embed_dim)
        # Zero initialization: scale=0, shift=0 at start
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: (B, S, D) input features.
            cond: (B, S, D_cond) per-position conditioning signal, or
                  (B, D_cond) single conditioning broadcast across positions.
        Returns:
            (B, S, D) modulated features.
        """
        # cond_proj produces scale and shift
        projected = self.cond_proj(cond)  # (B, [S,] 2*D)
        scale, shift = projected.chunk(2, dim=-1)
        if scale.dim() == 2:
            # Single conditioning vector: broadcast across sequence
            scale = scale.unsqueeze(1)  # (B, 1, D)
            shift = shift.unsqueeze(1)  # (B, 1, D)
        # At init: scale=0, shift=0, so output is 0.
        # With residual connection (x + AdaLN(x)), block starts as identity.
        return scale * self.norm(x) + shift


if __name__ == "__main__":
    adaln = AdaLN(embed_dim=192, cond_dim=192)

    # Test with (B, D_cond) conditioning (broadcast)
    x = torch.randn(4, 3, 192)
    cond = torch.randn(4, 192)
    out = adaln(x, cond)
    print(f"Input: {x.shape}, Cond: {cond.shape}, Output: {out.shape}")
    print(f"Output max abs (should be ~0 at init): {out.abs().max().item():.6f}")
    assert out.abs().max().item() < 1e-6, "AdaLN zero init failed (2D cond)!"

    # Test with (B, S, D_cond) per-position conditioning
    cond_3d = torch.randn(4, 3, 192)
    out_3d = adaln(x, cond_3d)
    print(f"Input: {x.shape}, Cond: {cond_3d.shape}, Output: {out_3d.shape}")
    print(f"Output max abs (should be ~0 at init): {out_3d.abs().max().item():.6f}")
    assert out_3d.abs().max().item() < 1e-6, "AdaLN zero init failed (3D cond)!"

    print("Zero initialization verified for both 2D and 3D conditioning.")
