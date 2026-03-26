"""Centralized configuration dataclasses for LeWM."""

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """ViT-Tiny encoder configuration."""
    model_name: str = "vit_tiny_patch16_224"  # base timm model; patch_size overridden to 14 in encoder
    patch_size: int = 14
    num_layers: int = 12
    num_heads: int = 3
    embed_dim: int = 192
    img_size: int = 224


@dataclass
class PredictorConfig:
    """Causal transformer predictor configuration."""
    num_layers: int = 6
    num_heads: int = 16
    embed_dim: int = 192
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    history_length: int = 3
    action_dim: int = 2
    max_seq_len: int = 16


@dataclass
class SIGRegConfig:
    """SIGReg anti-collapse regularizer configuration."""
    num_projections: int = 1024
    num_knots: int = 17
    t_min: float = 0.2
    t_max: float = 4.0
    loss_weight: float = 0.1


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    batch_size: int = 128
    sub_trajectory_length: int = 4
    frame_skip: int = 5
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    max_grad_norm: float = 1.0


@dataclass
class CEMConfig:
    """Cross-Entropy Method planner configuration."""
    num_samples: int = 300
    num_elites: int = 30
    num_iterations: int = 30
    horizon: int = 5
    sigma_init: float = 1.0


@dataclass
class LeWMConfig:
    """Top-level configuration aggregating all sub-configs."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cem: CEMConfig = field(default_factory=CEMConfig)


if __name__ == "__main__":
    config = LeWMConfig()
    print("LeWMConfig created successfully:")
    print(f"  Encoder: {config.encoder}")
    print(f"  Predictor: {config.predictor}")
    print(f"  SIGReg: {config.sigreg}")
    print(f"  Training: {config.training}")
    print(f"  CEM: {config.cem}")
