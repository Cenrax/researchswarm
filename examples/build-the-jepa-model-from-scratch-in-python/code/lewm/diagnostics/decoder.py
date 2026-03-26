"""Optional cross-attention transformer decoder for visualization."""

import torch
import torch.nn as nn


class LatentDecoder(nn.Module):
    """Cross-attention transformer decoder for reconstructing images.

    Used only for visualization and debugging -- not part of training
    or planning. Can be trained separately with a frozen encoder.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        img_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size
        self.patches_per_side = img_size // patch_size

        # Learnable query tokens (one per output patch)
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Project patches back to pixel space
        self.patch_proj = nn.Linear(
            embed_dim, 3 * patch_size * patch_size
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent embedding to image.

        Args:
            z: (B, D) latent embedding.
        Returns:
            img: (B, 3, img_size, img_size) reconstructed image.
        """
        B = z.shape[0]
        memory = z.unsqueeze(1)  # (B, 1, D)
        queries = self.query_tokens.expand(B, -1, -1) + self.pos_embed

        out = self.decoder(queries, memory)  # (B, num_patches, D)
        patches = self.patch_proj(out)  # (B, num_patches, 3*P*P)

        # Reshape to image
        P = self.patch_size
        H_p = W_p = self.patches_per_side
        img = patches.reshape(B, H_p, W_p, 3, P, P)
        img = img.permute(0, 3, 1, 4, 2, 5).reshape(
            B, 3, H_p * P, W_p * P
        )
        return img


if __name__ == "__main__":
    decoder = LatentDecoder(embed_dim=192, img_size=224, patch_size=16)
    z = torch.randn(2, 192)
    img = decoder(z)
    print(f"Input: {z.shape}, Output: {img.shape}")
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {params:,}")
