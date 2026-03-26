"""ViT-Tiny encoder using timm, extracts CLS token, applies projection head."""

import torch
import torch.nn as nn
import timm

from ..config import EncoderConfig
from .projection_head import ProjectionHead


class ViTEncoder(nn.Module):
    """ViT-Tiny encoder for LeWM.

    Architecture: ViT-Tiny (patch=14, 12 layers, 3 heads, dim=192)
    followed by a Linear + BatchNorm1d projection head.
    Approximately 5M parameters total.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Load ViT-Tiny from timm without pretrained weights.
        # num_classes=0 removes the classification head and returns
        # the CLS token after the final LayerNorm.
        self.vit = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=0,
            img_size=config.img_size,
            patch_size=config.patch_size,
        )
        self.projection = ProjectionHead(config.embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to latent embeddings.

        Args:
            obs: (B, 3, 224, 224) RGB images.
        Returns:
            z: (B, D) latent embeddings where D=embed_dim.
        """
        cls_token = self.vit(obs)  # (B, embed_dim)
        z = self.projection(cls_token)  # (B, embed_dim)
        return z

    def encode_trajectory(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of image trajectories.

        Args:
            obs: (B, T, 3, H, W) trajectory of RGB images.
        Returns:
            z: (B, T, D) trajectory of latent embeddings.
        """
        B, T, C, H, W = obs.shape
        z = self.forward(obs.reshape(B * T, C, H, W))  # (B*T, D)
        return z.reshape(B, T, -1)  # (B, T, D)


if __name__ == "__main__":
    config = EncoderConfig()
    encoder = ViTEncoder(config)

    # Test single image
    x = torch.randn(4, 3, 224, 224)
    z = encoder(x)
    print(f"Single image - Input: {x.shape}, Output: {z.shape}")

    # Test trajectory
    x_traj = torch.randn(2, 4, 3, 224, 224)
    z_traj = encoder.encode_trajectory(x_traj)
    print(f"Trajectory - Input: {x_traj.shape}, Output: {z_traj.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
