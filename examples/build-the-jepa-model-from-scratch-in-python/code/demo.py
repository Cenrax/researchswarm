"""Complete runnable demo of LeWM JEPA.

Creates synthetic trajectory data (random images + random actions),
instantiates the full model, runs a few training steps, then runs
CEM planning. Works out of the box with just `python demo.py`.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from lewm.config import LeWMConfig
from lewm.models.lewm import LeWM
from lewm.losses.sigreg import SIGReg, compute_sigreg_stepwise
from lewm.losses.prediction_loss import prediction_loss
from lewm.data.dataset import TrajectoryDataset, make_synthetic_dataset
from lewm.data.transforms import get_default_transform
from lewm.planning.cem import CEMPlanner
from lewm.planning.rollout import latent_rollout
from lewm.diagnostics.metrics import temporal_straightness, embedding_stats


def create_synthetic_data(
    num_trajectories: int = 5,
    traj_length: int = 30,
    img_size: int = 64,
    action_dim: int = 2,
) -> list[dict]:
    """Create synthetic trajectory data with smooth transitions.

    Each trajectory has random images and random actions.
    """
    print(f"Creating {num_trajectories} synthetic trajectories "
          f"(length={traj_length}, img={img_size}x{img_size})...")
    return make_synthetic_dataset(
        num_trajectories=num_trajectories,
        traj_length=traj_length,
        img_size=img_size,
        action_dim=action_dim,
    )


def run_training_steps(
    model: LeWM,
    dataloader: DataLoader,
    sigreg: SIGReg,
    num_steps: int = 5,
    lr: float = 3e-4,
    lambda_sigreg: float = 0.1,
    device: str = "cpu",
) -> None:
    """Run a few training steps and print losses.

    Args:
        model: LeWM model.
        dataloader: Training data loader.
        sigreg: SIGReg module.
        num_steps: Number of training steps to run.
        lr: Learning rate.
        lambda_sigreg: SIGReg loss weight.
        device: Device to use.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\nRunning {num_steps} training steps...")
    print("-" * 70)

    step = 0
    for batch in dataloader:
        if step >= num_steps:
            break

        obs = batch["obs"].to(device)  # (B, T, 3, 224, 224)
        actions = batch["actions"].to(device)  # (B, T, A)

        # Forward pass (teacher-forcing)
        z, z_hat = model(obs, actions)

        # Losses
        l_pred = prediction_loss(z, z_hat)
        l_sigreg = compute_sigreg_stepwise(z, sigreg)
        loss = l_pred + lambda_sigreg * l_sigreg

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print stats
        emb_stats = embedding_stats(z.detach())
        print(
            f"Step {step+1}/{num_steps} | "
            f"pred_loss: {l_pred.item():.4f} | "
            f"sigreg: {l_sigreg.item():.6f} | "
            f"total: {loss.item():.4f} | "
            f"emb_var: {emb_stats['variance']:.4f}"
        )
        step += 1

    print("-" * 70)
    print("Training steps complete.")


def run_cem_planning(
    model: LeWM,
    config: LeWMConfig,
    device: str = "cpu",
) -> None:
    """Run CEM planning with random initial and goal observations.

    Args:
        model: Trained LeWM model.
        config: Full configuration.
        device: Device to use.
    """
    model.eval()
    planner = CEMPlanner(model, config, device=device)

    print("\nRunning CEM planning...")
    print("-" * 70)

    # Create random observations for demo
    obs_init = torch.randn(1, 3, 224, 224).to(device)
    obs_goal = torch.randn(1, 3, 224, 224).to(device)

    # Encode them
    with torch.no_grad():
        z_init = model.encoder(obs_init)
        z_goal = model.encoder(obs_goal)
        init_dist = ((z_init - z_goal) ** 2).sum().item()
        print(f"Initial latent distance to goal: {init_dist:.4f}")

    # Plan
    best_actions = planner.plan(obs_init, obs_goal)
    print(f"Planned action sequence shape: {best_actions.shape}")
    print(f"Planned actions:\n{best_actions}")

    # Verify: rollout with planned actions
    with torch.no_grad():
        z_traj = latent_rollout(model.predictor, z_init, best_actions.unsqueeze(0))
        z_final = z_traj[:, -1, :]
        final_dist = ((z_final - z_goal) ** 2).sum().item()
        print(f"Final latent distance after rollout: {final_dist:.4f}")
        improvement = (init_dist - final_dist) / init_dist * 100
        print(f"Distance improvement: {improvement:.1f}%")

    print("-" * 70)
    print("CEM planning complete.")


def main() -> None:
    """Run the complete LeWM demo."""
    print("=" * 70)
    print("LeWM JEPA Demo")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Configuration (small for demo speed)
    config = LeWMConfig()
    config.training.batch_size = 4
    config.training.num_epochs = 1
    config.sigreg.num_projections = 128  # reduced for CPU speed
    config.cem.num_samples = 20
    config.cem.num_elites = 5
    config.cem.num_iterations = 5
    config.cem.horizon = 3

    # Create model
    print("\n[1] Creating LeWM model...")
    model = LeWM(config.encoder, config.predictor).to(device)
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    pred_params = sum(p.numel() for p in model.predictor.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Encoder: {enc_params:,} params (~{enc_params/1e6:.1f}M)")
    print(f"  Predictor: {pred_params:,} params (~{pred_params/1e6:.1f}M)")
    print(f"  Total: {total_params:,} params (~{total_params/1e6:.1f}M)")

    # Create SIGReg
    sigreg = SIGReg(
        embed_dim=config.encoder.embed_dim,
        num_projections=config.sigreg.num_projections,
        num_knots=config.sigreg.num_knots,
    ).to(device)

    # Create synthetic dataset
    print("\n[2] Creating synthetic dataset...")
    trajs = create_synthetic_data(
        num_trajectories=5, traj_length=30, img_size=64, action_dim=2
    )
    transform = get_default_transform(img_size=224)
    dataset = TrajectoryDataset(
        trajs, sub_traj_length=4, frame_skip=5, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=True
    )
    print(f"  Sub-trajectories: {len(dataset)}")
    sample = dataset[0]
    print(f"  Sample obs shape: {sample['obs'].shape}")
    print(f"  Sample actions shape: {sample['actions'].shape}")

    # Quick forward pass test
    print("\n[3] Verifying forward pass...")
    model.train()
    batch = next(iter(loader))
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    z, z_hat = model(obs, actions)
    print(f"  z shape: {z.shape}")
    print(f"  z_hat shape: {z_hat.shape}")
    l_pred = prediction_loss(z, z_hat)
    l_sigreg = compute_sigreg_stepwise(z, sigreg)
    print(f"  pred_loss: {l_pred.item():.4f}")
    print(f"  sigreg_loss: {l_sigreg.item():.6f}")

    # Verify backward pass
    loss = l_pred + config.sigreg.loss_weight * l_sigreg
    loss.backward()
    enc_grad_ok = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    pred_grad_ok = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.predictor.parameters()
    )
    print(f"  Gradients flow to encoder: {enc_grad_ok}")
    print(f"  Gradients flow to predictor: {pred_grad_ok}")
    model.zero_grad()

    # Training steps
    print("\n[4] Training...")
    run_training_steps(
        model, loader, sigreg,
        num_steps=3,
        lr=config.training.learning_rate,
        lambda_sigreg=config.sigreg.loss_weight,
        device=device,
    )

    # Diagnostics
    print("\n[5] Diagnostics...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        obs = batch["obs"].to(device)
        z = model.encoder.encode_trajectory(obs)
        stats = embedding_stats(z)
        print(f"  Embedding stats: {stats}")
        if z.shape[1] >= 3:
            s = temporal_straightness(z)
            print(f"  Temporal straightness: {s:.4f}")

    # CEM Planning
    print("\n[6] CEM Planning...")
    run_cem_planning(model, config, device=device)

    print("\n" + "=" * 70)
    print("Demo complete. All components working.")
    print("=" * 70)


if __name__ == "__main__":
    main()
