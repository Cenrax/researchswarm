"""Combined LeWM model: encoder + predictor."""

import torch
import torch.nn as nn

from ..config import EncoderConfig, PredictorConfig
from .encoder import ViTEncoder
from .predictor import CausalPredictor


class LeWM(nn.Module):
    """LeWorldModel: Joint Embedding Predictive Architecture.

    Combines a ViT-Tiny encoder and a causal transformer predictor.
    Supports teacher-forcing training and autoregressive rollout.
    No EMA, no stop-gradient -- everything is end-to-end differentiable.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        predictor_config: PredictorConfig,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(encoder_config)
        self.predictor = CausalPredictor(predictor_config)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass with teacher-forcing.

        Encodes ALL frames with the encoder, then runs the predictor
        on ground-truth embeddings (not autoregressive predictions).

        Args:
            obs: (B, T, 3, 224, 224) trajectory of RGB images.
            actions: (B, T, A) actions for each timestep.
        Returns:
            z: (B, T, D) encoder embeddings (ground truth targets).
            z_hat: (B, T, D) predictor outputs.
        """
        z = self.encoder.encode_trajectory(obs)  # (B, T, D)
        z_hat = self.predictor(z, actions)  # (B, T, D)
        return z, z_hat

    @torch.no_grad()
    def rollout(
        self, obs_init: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Autoregressive rollout for planning.

        Encodes initial observation, then predicts future states
        step-by-step using previous predictions (not ground truth).

        Args:
            obs_init: (B, 3, 224, 224) initial observation.
            actions: (B, H, A) action sequence over planning horizon.
        Returns:
            z_traj: (B, H+1, D) predicted latent trajectory.
        """
        B = obs_init.shape[0]
        H = actions.shape[1]

        z_init = self.encoder(obs_init)  # (B, D)
        z_traj = [z_init]

        for t in range(H):
            # Build history window (up to history_length)
            hist_len = min(len(z_traj), self.predictor.history_length)
            history = torch.stack(z_traj[-hist_len:], dim=1)  # (B, N, D)

            # Current action
            a_t = actions[:, t: t + 1, :]  # (B, 1, A)
            # Expand action to match history length
            a_expand = a_t.expand(-1, history.shape[1], -1)

            z_hat = self.predictor(history, a_expand)
            z_next = z_hat[:, -1, :]  # (B, D) take last prediction
            z_traj.append(z_next)

        return torch.stack(z_traj, dim=1)  # (B, H+1, D)

    @torch.no_grad()
    def rollout_from_embedding(
        self, z_init: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Rollout from a pre-computed initial embedding.

        Args:
            z_init: (B, D) initial latent embedding.
            actions: (B, H, A) action sequence.
        Returns:
            z_traj: (B, H+1, D) predicted latent trajectory.
        """
        H = actions.shape[1]
        z_traj = [z_init]

        for t in range(H):
            hist_len = min(len(z_traj), self.predictor.history_length)
            history = torch.stack(z_traj[-hist_len:], dim=1)
            a_t = actions[:, t: t + 1, :]
            a_expand = a_t.expand(-1, history.shape[1], -1)

            z_hat = self.predictor(history, a_expand)
            z_next = z_hat[:, -1, :]
            z_traj.append(z_next)

        return torch.stack(z_traj, dim=1)


if __name__ == "__main__":
    from ..config import EncoderConfig, PredictorConfig

    enc_cfg = EncoderConfig()
    pred_cfg = PredictorConfig()
    model = LeWM(enc_cfg, pred_cfg)

    # Training forward
    B, T, A = 2, 4, 2
    obs = torch.randn(B, T, 3, 224, 224)
    actions = torch.randn(B, T, A)
    z, z_hat = model(obs, actions)
    print(f"Training - z: {z.shape}, z_hat: {z_hat.shape}")

    # Total params
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,} (~{total/1e6:.1f}M)")
