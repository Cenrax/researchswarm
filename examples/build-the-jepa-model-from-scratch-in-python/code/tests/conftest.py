"""Shared fixtures for LeWM tests."""

import pytest
import torch

from lewm.config import (
    EncoderConfig,
    PredictorConfig,
    SIGRegConfig,
    TrainingConfig,
    CEMConfig,
    LeWMConfig,
)


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def encoder_config() -> EncoderConfig:
    return EncoderConfig()


@pytest.fixture
def predictor_config() -> PredictorConfig:
    return PredictorConfig()


@pytest.fixture
def small_config() -> LeWMConfig:
    """Config with reduced sizes for fast testing."""
    config = LeWMConfig()
    config.training.batch_size = 4
    config.training.num_epochs = 1
    config.sigreg.num_projections = 64
    config.cem.num_samples = 10
    config.cem.num_elites = 3
    config.cem.num_iterations = 3
    config.cem.horizon = 2
    return config


@pytest.fixture
def sample_images() -> torch.Tensor:
    """Small batch of random images: (B, 3, 224, 224)."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_trajectory() -> tuple[torch.Tensor, torch.Tensor]:
    """Sample trajectory: obs (B, T, 3, 224, 224), actions (B, T, A)."""
    B, T, A = 2, 4, 2
    obs = torch.randn(B, T, 3, 224, 224)
    actions = torch.randn(B, T, A)
    return obs, actions


@pytest.fixture
def sample_embeddings() -> torch.Tensor:
    """Sample embeddings: (B, D)."""
    return torch.randn(32, 192)


@pytest.fixture
def sample_trajectory_embeddings() -> torch.Tensor:
    """Sample trajectory embeddings: (B, T, D)."""
    return torch.randn(8, 4, 192)
