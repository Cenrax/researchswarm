"""Autoregressive latent rollout using predictor."""

import torch

from ..models.predictor import CausalPredictor


@torch.no_grad()
def latent_rollout(
    predictor: CausalPredictor,
    z_init: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Autoregressive latent rollout using the predictor.

    Starts from z_init and predicts forward H steps using the given actions.

    Args:
        predictor: Trained CausalPredictor in eval mode.
        z_init: (B, D) initial latent embedding.
        actions: (B, H, A) action sequence over planning horizon.
    Returns:
        z_traj: (B, H+1, D) predicted latent trajectory including z_init.
    """
    H = actions.shape[1]
    history_len = predictor.history_length
    z_traj = [z_init]

    for t in range(H):
        hist_len = min(len(z_traj), history_len)
        history = torch.stack(z_traj[-hist_len:], dim=1)  # (B, N, D)

        a_t = actions[:, t: t + 1, :]  # (B, 1, A)
        a_expand = a_t.expand(-1, history.shape[1], -1)

        z_hat = predictor(history, a_expand)
        z_next = z_hat[:, -1, :]  # (B, D)
        z_traj.append(z_next)

    return torch.stack(z_traj, dim=1)  # (B, H+1, D)


@torch.no_grad()
def batch_latent_rollout(
    predictor: CausalPredictor,
    z_init: torch.Tensor,
    actions_batch: torch.Tensor,
) -> torch.Tensor:
    """Batch rollout for CEM: run N candidate action sequences in parallel.

    Args:
        predictor: Trained CausalPredictor in eval mode.
        z_init: (B, D) initial latent (will be expanded to N samples).
        actions_batch: (N, H, A) batch of candidate action sequences.
    Returns:
        z_trajs: (N, H+1, D) batch of predicted trajectories.
    """
    return latent_rollout(predictor, z_init, actions_batch)


if __name__ == "__main__":
    from lewm.config import PredictorConfig
    from lewm.models.predictor import CausalPredictor

    config = PredictorConfig()
    predictor = CausalPredictor(config)
    predictor.eval()

    B, H, D, A = 4, 5, 192, 2
    z_init = torch.randn(B, D)
    actions = torch.randn(B, H, A)

    z_traj = latent_rollout(predictor, z_init, actions)
    print(f"Rollout - z_init: {z_init.shape}, actions: {actions.shape}")
    print(f"z_traj: {z_traj.shape}")
    assert z_traj.shape == (B, H + 1, D)
    print("Rollout test passed.")
