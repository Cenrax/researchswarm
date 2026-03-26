"""MSE prediction loss between predicted and target embeddings."""

import torch
import torch.nn.functional as F


def prediction_loss(z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    """Teacher-forcing MSE prediction loss.

    The prediction at position t should match the embedding at position t+1.
    z_hat[:, t] predicts z[:, t+1].

    Args:
        z: (B, T, D) ground-truth encoder embeddings.
        z_hat: (B, T, D) predicted embeddings from predictor.
    Returns:
        Scalar MSE loss.
    """
    target = z[:, 1:, :]  # (B, T-1, D)
    pred = z_hat[:, :-1, :]  # (B, T-1, D)
    return F.mse_loss(pred, target)


if __name__ == "__main__":
    z = torch.randn(4, 4, 192)
    z_hat = torch.randn(4, 4, 192)
    loss = prediction_loss(z, z_hat)
    print(f"Prediction loss: {loss.item():.4f}")

    # When predictions are perfect, loss should be 0
    z_hat_perfect = z.clone()
    z_hat_perfect[:, :-1, :] = z[:, 1:, :]
    loss_perfect = prediction_loss(z, z_hat_perfect)
    print(f"Perfect prediction loss: {loss_perfect.item():.6f} (should be 0)")
