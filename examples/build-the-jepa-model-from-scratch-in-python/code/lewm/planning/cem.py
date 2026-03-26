"""Cross-Entropy Method planner for goal-conditioned latent planning."""

from typing import Optional

import torch

from ..config import LeWMConfig
from ..models.lewm import LeWM
from .rollout import latent_rollout


class CEMPlanner:
    """Cross-Entropy Method planner.

    Optimizes action sequences in latent space to minimize distance
    to a goal embedding. Uses 300 samples, 30 iterations, top-30 elites.
    """

    def __init__(
        self,
        model: LeWM,
        config: LeWMConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def plan(
        self,
        obs_init: torch.Tensor,
        obs_goal: torch.Tensor,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plan an action sequence using CEM.

        Args:
            obs_init: (3, H, W) or (1, 3, H, W) initial observation.
            obs_goal: (3, H, W) or (1, 3, H, W) goal observation.
            action_low: (A,) lower bounds for actions (optional).
            action_high: (A,) upper bounds for actions (optional).
        Returns:
            best_actions: (H, A) optimized action sequence.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            cem = self.config.cem
            A = self.config.predictor.action_dim
            H = cem.horizon
            N = cem.num_samples
            K = cem.num_elites

            # Ensure batch dimension
            if obs_init.dim() == 3:
                obs_init = obs_init.unsqueeze(0)
            if obs_goal.dim() == 3:
                obs_goal = obs_goal.unsqueeze(0)

            obs_init = obs_init.to(self.device)
            obs_goal = obs_goal.to(self.device)

            # Encode initial and goal observations
            z_init = self.model.encoder(obs_init)  # (1, D)
            z_goal = self.model.encoder(obs_goal)  # (1, D)

            # Initialize CEM distribution
            mu = torch.zeros(H, A, device=self.device)
            sigma = torch.ones(H, A, device=self.device) * cem.sigma_init

            for iteration in range(cem.num_iterations):
                # Sample action sequences
                noise = torch.randn(N, H, A, device=self.device)
                actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

                # Clip to action bounds
                if action_low is not None and action_high is not None:
                    actions = actions.clamp(
                        action_low.to(self.device),
                        action_high.to(self.device),
                    )

                # Expand z_init to (N, D) for batch rollout
                z_init_expanded = z_init.expand(N, -1)

                # Rollout each action sequence
                z_traj = latent_rollout(
                    self.model.predictor, z_init_expanded, actions
                )  # (N, H+1, D)

                # Cost: L2 distance at final step to goal
                z_final = z_traj[:, -1, :]  # (N, D)
                costs = ((z_final - z_goal) ** 2).sum(dim=-1)  # (N,)

                # Select elites (lowest cost)
                elite_idx = costs.argsort()[:K]
                elite_actions = actions[elite_idx]  # (K, H, A)

                # Update distribution
                mu = elite_actions.mean(dim=0)  # (H, A)
                sigma = elite_actions.std(dim=0).clamp(min=1e-4)  # (H, A)

            return mu  # (H, A) best action sequence
        finally:
            self.model.train(was_training)


if __name__ == "__main__":
    from lewm.config import LeWMConfig

    config = LeWMConfig()
    # Use small CEM for testing
    config.cem.num_samples = 10
    config.cem.num_elites = 3
    config.cem.num_iterations = 5
    config.cem.horizon = 3

    model = LeWM(config.encoder, config.predictor)
    planner = CEMPlanner(model, config, device="cpu")

    obs_init = torch.randn(1, 3, 224, 224)
    obs_goal = torch.randn(1, 3, 224, 224)

    best_actions = planner.plan(obs_init, obs_goal)
    print(f"CEM output: {best_actions.shape}")
    print(f"Action values: {best_actions}")
