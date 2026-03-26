"""Training loop: joint optimization of encoder+predictor with MSE + SIGReg."""

import torch
from torch.utils.data import DataLoader

from ..config import LeWMConfig
from ..models.lewm import LeWM
from ..losses.sigreg import SIGReg, compute_sigreg_stepwise
from ..losses.prediction_loss import prediction_loss
from .logger import TrainingLogger


class Trainer:
    """LeWM trainer with joint encoder+predictor optimization.

    Uses MSE prediction loss + lambda * SIGReg regularizer.
    No EMA, no stop-gradient -- everything is end-to-end differentiable.
    """

    def __init__(
        self,
        model: LeWM,
        config: LeWMConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.sigreg = SIGReg(
            embed_dim=config.encoder.embed_dim,
            num_projections=config.sigreg.num_projections,
            num_knots=config.sigreg.num_knots,
            t_min=config.sigreg.t_min,
            t_max=config.sigreg.t_max,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.logger = TrainingLogger(log_interval=1)
        self.lambda_sigreg = config.sigreg.loss_weight

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding batches with 'obs' and 'actions'.
        Returns:
            Dictionary of averaged loss statistics.
        """
        self.model.train()
        stats = {"pred_loss": 0.0, "sigreg_loss": 0.0, "total_loss": 0.0, "steps": 0}

        for batch in dataloader:
            obs = batch["obs"].to(self.device)  # (B, T, 3, H, W)
            actions = batch["actions"].to(self.device)  # (B, T, A)

            # Forward pass (teacher-forcing)
            z, z_hat = self.model(obs, actions)

            # Prediction loss
            l_pred = prediction_loss(z, z_hat)

            # SIGReg regularizer (step-wise per timestep)
            l_sigreg = compute_sigreg_stepwise(z, self.sigreg)

            # Total loss
            loss = l_pred + self.lambda_sigreg * l_sigreg

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.max_grad_norm,
            )
            self.optimizer.step()

            # Accumulate stats
            stats["pred_loss"] += l_pred.item()
            stats["sigreg_loss"] += l_sigreg.item()
            stats["total_loss"] += loss.item()
            stats["steps"] += 1

        # Average over steps
        n = max(stats["steps"], 1)
        return {
            "pred_loss": stats["pred_loss"] / n,
            "sigreg_loss": stats["sigreg_loss"] / n,
            "total_loss": stats["total_loss"] / n,
        }

    def train(self, dataloader: DataLoader) -> list[dict]:
        """Full training loop.

        Args:
            dataloader: DataLoader yielding batches.
        Returns:
            List of per-epoch statistics.
        """
        self.logger.start()
        history = []

        for epoch in range(1, self.config.training.num_epochs + 1):
            stats = self.train_epoch(dataloader)
            self.logger.log_epoch(
                epoch, self.config.training.num_epochs, stats
            )
            history.append(stats)

            # Monitor for collapse
            self._check_collapse(dataloader)

        return history

    @torch.no_grad()
    def _check_collapse(self, dataloader: DataLoader) -> None:
        """Check if embeddings have collapsed by measuring variance."""
        self.model.eval()
        try:
            batch = next(iter(dataloader))
            obs = batch["obs"].to(self.device)
            z = self.model.encoder.encode_trajectory(obs)
            var = z.var(dim=0).mean().item()
            if var < 1e-4:
                self.logger.log_warning(
                    f"Potential collapse detected! Embedding variance = {var:.6f}"
                )
        except StopIteration:
            pass
        self.model.train()


if __name__ == "__main__":
    from lewm.config import LeWMConfig
    from lewm.models.lewm import LeWM
    from lewm.data.dataset import TrajectoryDataset, make_synthetic_dataset
    from lewm.data.transforms import get_default_transform

    config = LeWMConfig()
    config.training.num_epochs = 2
    config.training.batch_size = 2
    config.sigreg.num_projections = 64  # fewer for speed

    model = LeWM(config.encoder, config.predictor)
    trainer = Trainer(model, config, device="cpu")

    trajs = make_synthetic_dataset(num_trajectories=3, traj_length=30)
    transform = get_default_transform()
    dataset = TrajectoryDataset(trajs, sub_traj_length=4, frame_skip=5, transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    history = trainer.train(loader)
    print(f"Training complete. {len(history)} epochs.")
