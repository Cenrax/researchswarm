"""Diagnostic metrics for LeWM training analysis."""

import torch
import torch.nn.functional as F


def temporal_straightness(z: torch.Tensor) -> float:
    """Compute temporal straightness metric (Eq. 9).

    Measures how straight latent trajectories are by computing
    the average cosine similarity between consecutive velocity vectors.
    A value near 1 indicates straight trajectories.

    Args:
        z: (B, T, D) latent trajectories where T >= 3.
    Returns:
        Average cosine similarity in [-1, 1].
    """
    if z.shape[1] < 3:
        return 0.0

    v = z[:, 1:, :] - z[:, :-1, :]  # (B, T-1, D) velocity vectors
    v_t = v[:, :-1, :]  # (B, T-2, D)
    v_tp1 = v[:, 1:, :]  # (B, T-2, D)

    cos_sim = F.cosine_similarity(v_t, v_tp1, dim=-1)  # (B, T-2)
    return cos_sim.mean().item()


def embedding_stats(z: torch.Tensor) -> dict[str, float]:
    """Compute embedding distribution statistics for monitoring.

    Useful for detecting representation collapse (low variance)
    or degenerate distributions.

    Args:
        z: (B, D) or (B, T, D) embeddings.
    Returns:
        Dictionary of statistics.
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.shape[-1])

    return {
        "mean_norm": z.norm(dim=-1).mean().item(),
        "std_per_dim": z.std(dim=0).mean().item(),
        "mean_per_dim": z.mean(dim=0).abs().mean().item(),
        "max_abs": z.abs().max().item(),
        "variance": z.var().item(),
    }


if __name__ == "__main__":
    # Straight trajectory test
    z_straight = torch.zeros(4, 5, 192)
    for t in range(5):
        z_straight[:, t, :] = t * torch.randn(4, 192)
    s = temporal_straightness(z_straight)
    print(f"Temporal straightness (linear trajectory): {s:.4f}")

    # Random trajectory
    z_random = torch.randn(4, 5, 192)
    s_random = temporal_straightness(z_random)
    print(f"Temporal straightness (random): {s_random:.4f}")

    # Embedding stats
    z = torch.randn(32, 192)
    stats = embedding_stats(z)
    print(f"Embedding stats: {stats}")
