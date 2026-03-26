"""Tests for SIGReg regularizer."""

import torch
import pytest

from lewm.losses.sigreg import SIGReg, compute_sigreg_stepwise


@pytest.fixture
def sigreg() -> SIGReg:
    return SIGReg(embed_dim=192, num_projections=256, num_knots=17)


class TestSIGReg:
    def test_gaussian_input_low_loss(self, sigreg: SIGReg) -> None:
        """SIGReg of standard Gaussian samples should be near zero."""
        torch.manual_seed(42)
        Z = torch.randn(128, 192)
        loss = sigreg(Z)
        assert loss.item() < 0.05, f"SIGReg(Gaussian) = {loss.item()}, expected < 0.05"

    def test_collapsed_input_high_loss(self, sigreg: SIGReg) -> None:
        """SIGReg of near-zero (collapsed) embeddings should be large."""
        Z = torch.zeros(128, 192) + 0.001 * torch.randn(128, 192)
        loss = sigreg(Z)
        assert loss.item() > 0.05, f"SIGReg(collapsed) = {loss.item()}, expected > 0.05"

    def test_gradient_flows(self, sigreg: SIGReg) -> None:
        """Gradients should flow back through SIGReg."""
        Z = torch.randn(32, 192, requires_grad=True)
        loss = sigreg(Z)
        loss.backward()
        assert Z.grad is not None, "No gradient on input"
        assert not torch.isnan(Z.grad).any(), "NaN in gradient"
        assert Z.grad.abs().sum() > 0, "Zero gradient"

    def test_output_is_scalar(self, sigreg: SIGReg) -> None:
        """SIGReg should return a scalar."""
        Z = torch.randn(16, 192)
        loss = sigreg(Z)
        assert loss.dim() == 0, f"Expected scalar, got dim={loss.dim()}"

    def test_non_negative(self, sigreg: SIGReg) -> None:
        """SIGReg should be non-negative (it's an integral of squares)."""
        Z = torch.randn(32, 192)
        loss = sigreg(Z)
        assert loss.item() >= 0, f"SIGReg = {loss.item()}, expected >= 0"

    def test_stepwise_application(self, sigreg: SIGReg) -> None:
        """Step-wise SIGReg should produce a scalar for trajectory input."""
        embeddings = torch.randn(8, 4, 192)
        loss = compute_sigreg_stepwise(embeddings, sigreg)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same SIGReg value."""
        sigreg = SIGReg(embed_dim=64, num_projections=64)
        Z = torch.randn(16, 64)
        torch.manual_seed(123)
        v1 = sigreg(Z)
        torch.manual_seed(123)
        v2 = sigreg(Z)
        assert torch.allclose(v1, v2), "SIGReg not deterministic with same seed"
