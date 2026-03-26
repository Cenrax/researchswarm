"""Tests for causal transformer predictor."""

import torch
import pytest

from lewm.config import PredictorConfig
from lewm.models.predictor import CausalPredictor


@pytest.fixture
def predictor() -> CausalPredictor:
    return CausalPredictor(PredictorConfig())


class TestPredictor:
    def test_output_shape(self, predictor: CausalPredictor) -> None:
        """Predictor output should match input shape."""
        B, T, D, A = 4, 4, 192, 2
        z = torch.randn(B, T, D)
        actions = torch.randn(B, T, A)
        z_hat = predictor(z, actions)
        assert z_hat.shape == (B, T, D)

    def test_causal_mask_attention_only(self, predictor: CausalPredictor) -> None:
        """Changing future inputs should not affect past outputs when actions are fixed.

        We test with identical action inputs to isolate the causal mask effect.
        """
        B, T, D, A = 4, 4, 192, 2
        torch.manual_seed(42)
        z = torch.randn(B, T, D)
        # Use constant actions so mean pooling is unaffected
        actions = torch.ones(B, T, A)

        predictor.eval()
        with torch.no_grad():
            z_hat1 = predictor(z, actions)

            z2 = z.clone()
            z2[:, -1, :] = torch.randn(B, D)
            z_hat2 = predictor(z2, actions)

        # Position 0 should be identical (causal: cannot see position 3)
        diff = (z_hat1[:, 0, :] - z_hat2[:, 0, :]).abs().max().item()
        assert diff < 1e-5, f"Causal mask failed: diff at pos 0 = {diff}"

    def test_adaln_zero_init(self) -> None:
        """At initialization, AdaLN output should be near zero."""
        from lewm.models.adaln import AdaLN
        adaln = AdaLN(192, 192)
        x = torch.randn(4, 3, 192)
        # Test with (B, D_cond) broadcast conditioning
        cond_2d = torch.randn(4, 192)
        out_2d = adaln(x, cond_2d)
        assert out_2d.abs().max().item() < 1e-6, "AdaLN zero init failed (2D)"
        # Test with (B, S, D_cond) per-position conditioning
        cond_3d = torch.randn(4, 3, 192)
        out_3d = adaln(x, cond_3d)
        assert out_3d.abs().max().item() < 1e-6, "AdaLN zero init failed (3D)"

    def test_gradient_flow(self, predictor: CausalPredictor) -> None:
        """Gradients should flow through z and actions."""
        B, T, D, A = 2, 3, 192, 2
        z = torch.randn(B, T, D, requires_grad=True)
        actions = torch.randn(B, T, A, requires_grad=True)
        z_hat = predictor(z, actions)
        loss = z_hat.sum()
        loss.backward()
        assert z.grad is not None
        assert actions.grad is not None

    def test_variable_sequence_length(self, predictor: CausalPredictor) -> None:
        """Predictor should handle different sequence lengths."""
        B, D, A = 2, 192, 2
        for T in [1, 2, 4, 8]:
            z = torch.randn(B, T, D)
            actions = torch.randn(B, T, A)
            z_hat = predictor(z, actions)
            assert z_hat.shape == (B, T, D)

    def test_predict_step(self, predictor: CausalPredictor) -> None:
        """Single-step prediction should return (B, D)."""
        B, D, A = 4, 192, 2
        z_history = torch.randn(B, 3, D)
        action = torch.randn(B, 1, A)
        z_next = predictor.predict_step(z_history, action)
        assert z_next.shape == (B, D)
