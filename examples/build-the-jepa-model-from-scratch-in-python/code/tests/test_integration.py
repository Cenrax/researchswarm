"""Integration tests for the full LeWM pipeline."""

import torch
import pytest

from lewm.config import LeWMConfig
from lewm.models.lewm import LeWM
from lewm.losses.sigreg import SIGReg, compute_sigreg_stepwise
from lewm.losses.prediction_loss import prediction_loss


@pytest.fixture
def config() -> LeWMConfig:
    config = LeWMConfig()
    config.sigreg.num_projections = 64
    return config


@pytest.fixture
def model(config: LeWMConfig) -> LeWM:
    return LeWM(config.encoder, config.predictor)


class TestIntegration:
    def test_full_forward_pass(self, model: LeWM) -> None:
        """Full forward pass should produce correct shapes without errors."""
        B, T, A = 2, 4, 2
        obs = torch.randn(B, T, 3, 224, 224)
        actions = torch.randn(B, T, A)
        z, z_hat = model(obs, actions)
        assert z.shape == (B, T, 192)
        assert z_hat.shape == (B, T, 192)

    def test_backward_pass(self, model: LeWM, config: LeWMConfig) -> None:
        """Loss backward should propagate to both encoder and predictor."""
        B, T, A = 2, 4, 2
        obs = torch.randn(B, T, 3, 224, 224)
        actions = torch.randn(B, T, A)

        z, z_hat = model(obs, actions)
        sigreg = SIGReg(
            embed_dim=config.encoder.embed_dim,
            num_projections=config.sigreg.num_projections,
        )
        l_pred = prediction_loss(z, z_hat)
        l_sigreg = compute_sigreg_stepwise(z, sigreg)
        loss = l_pred + 0.1 * l_sigreg
        loss.backward()

        enc_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        pred_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.predictor.parameters()
        )
        assert enc_has_grad, "No gradients in encoder"
        assert pred_has_grad, "No gradients in predictor"

    def test_single_training_step(self, model: LeWM, config: LeWMConfig) -> None:
        """One training step should reduce loss on same input."""
        B, T, A = 2, 4, 2
        obs = torch.randn(B, T, 3, 224, 224)
        actions = torch.randn(B, T, A)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sigreg = SIGReg(
            embed_dim=config.encoder.embed_dim,
            num_projections=config.sigreg.num_projections,
        )

        # Step 1
        model.train()
        z, z_hat = model(obs, actions)
        l1 = prediction_loss(z, z_hat)
        loss = l1 + 0.1 * compute_sigreg_stepwise(z, sigreg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1_val = l1.item()

        # Step 2 (same data)
        z, z_hat = model(obs, actions)
        l2 = prediction_loss(z, z_hat)
        loss2_val = l2.item()

        # Loss should generally decrease or at least not explode
        assert loss2_val < loss1_val * 2, (
            f"Loss increased too much: {loss1_val:.4f} -> {loss2_val:.4f}"
        )

    def test_no_collapse_after_steps(self, model: LeWM, config: LeWMConfig) -> None:
        """After a few steps, embedding variance should remain positive."""
        B, T, A = 4, 4, 2
        obs = torch.randn(B, T, 3, 224, 224)
        actions = torch.randn(B, T, A)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sigreg = SIGReg(
            embed_dim=config.encoder.embed_dim,
            num_projections=config.sigreg.num_projections,
        )

        model.train()
        for _ in range(3):
            z, z_hat = model(obs, actions)
            loss = prediction_loss(z, z_hat) + 0.1 * compute_sigreg_stepwise(z, sigreg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            z, _ = model(obs, actions)
            var = z.var(dim=0).mean().item()
        assert var > 1e-6, f"Possible collapse: variance = {var}"

    def test_rollout_shape(self, model: LeWM) -> None:
        """Rollout should produce correct trajectory shape."""
        B, H, A = 2, 3, 2
        obs_init = torch.randn(B, 3, 224, 224)
        actions = torch.randn(B, H, A)

        model.eval()
        z_traj = model.rollout(obs_init, actions)
        assert z_traj.shape == (B, H + 1, 192)
