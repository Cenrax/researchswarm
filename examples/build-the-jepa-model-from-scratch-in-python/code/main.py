"""Main entry point tying all LeWM components together."""

import torch
from torch.utils.data import DataLoader

from lewm.config import LeWMConfig
from lewm.models.lewm import LeWM
from lewm.data.dataset import TrajectoryDataset, make_synthetic_dataset
from lewm.data.transforms import get_default_transform
from lewm.training.trainer import Trainer
from lewm.planning.cem import CEMPlanner
from lewm.diagnostics.metrics import temporal_straightness, embedding_stats


def main() -> None:
    """Run a complete LeWM pipeline: train on synthetic data, then plan."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configuration
    config = LeWMConfig()
    config.training.num_epochs = 2
    config.training.batch_size = 4
    config.sigreg.num_projections = 128  # fewer for speed on CPU
    config.cem.num_samples = 20
    config.cem.num_elites = 5
    config.cem.num_iterations = 5
    config.cem.horizon = 3

    # Create model
    model = LeWM(config.encoder, config.predictor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters (~{total_params/1e6:.1f}M)")

    # Create synthetic dataset
    trajs = make_synthetic_dataset(
        num_trajectories=5, traj_length=30, img_size=64, action_dim=2
    )
    transform = get_default_transform(img_size=224)
    dataset = TrajectoryDataset(
        trajs, sub_traj_length=4, frame_skip=5, transform=transform
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    print(f"Dataset: {len(dataset)} sub-trajectories")

    # Train
    print("\n--- Training ---")
    trainer = Trainer(model, config, device=device)
    history = trainer.train(loader)

    # Diagnostic metrics
    print("\n--- Diagnostics ---")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        obs = batch["obs"].to(device)
        z = model.encoder.encode_trajectory(obs)
        print(f"Embedding stats: {embedding_stats(z)}")
        print(f"Temporal straightness: {temporal_straightness(z):.4f}")

    # Planning
    print("\n--- CEM Planning ---")
    planner = CEMPlanner(model, config, device=device)
    obs_init = torch.randn(1, 3, 224, 224)
    obs_goal = torch.randn(1, 3, 224, 224)
    best_actions = planner.plan(obs_init, obs_goal)
    print(f"Planned actions shape: {best_actions.shape}")
    print(f"Planned actions:\n{best_actions}")

    print("\nDone.")


if __name__ == "__main__":
    main()
