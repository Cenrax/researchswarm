"""Tests for ViT-Tiny encoder."""

import torch
import torch.nn as nn
import pytest

from lewm.config import EncoderConfig
from lewm.models.encoder import ViTEncoder


@pytest.fixture
def encoder() -> ViTEncoder:
    return ViTEncoder(EncoderConfig())


class TestEncoder:
    def test_output_shape(self, encoder: ViTEncoder) -> None:
        """Single image batch should produce (B, D) output."""
        x = torch.randn(4, 3, 224, 224)
        z = encoder(x)
        assert z.shape == (4, 192)

    def test_trajectory_encoding(self, encoder: ViTEncoder) -> None:
        """Trajectory input should produce (B, T, D) output."""
        x = torch.randn(2, 4, 3, 224, 224)
        z = encoder.encode_trajectory(x)
        assert z.shape == (2, 4, 192)

    def test_parameter_count(self, encoder: ViTEncoder) -> None:
        """Encoder should have approximately 5M parameters."""
        total = sum(p.numel() for p in encoder.parameters())
        assert 3e6 < total < 8e6, f"Parameter count {total} out of expected range"

    def test_batchnorm_in_head(self, encoder: ViTEncoder) -> None:
        """Projection head must contain BatchNorm1d, not LayerNorm."""
        has_bn = any(
            isinstance(m, nn.BatchNorm1d)
            for m in encoder.projection.modules()
        )
        has_ln = any(
            isinstance(m, nn.LayerNorm)
            for m in encoder.projection.modules()
        )
        assert has_bn, "Projection head missing BatchNorm1d"
        assert not has_ln, "Projection head should not have LayerNorm"

    def test_gradient_flow(self, encoder: ViTEncoder) -> None:
        """Gradients should flow through the encoder."""
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        z = encoder(x)
        loss = z.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
