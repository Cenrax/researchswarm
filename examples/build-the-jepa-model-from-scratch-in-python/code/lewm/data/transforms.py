"""Image preprocessing transforms for LeWM."""

from torchvision import transforms


def get_default_transform(img_size: int = 224) -> transforms.Compose:
    """Get the default image preprocessing pipeline.

    Resizes to img_size x img_size, converts to tensor, and normalizes
    with ImageNet statistics.

    Args:
        img_size: Target image size (default 224).
    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_tensor_transform(img_size: int = 224) -> transforms.Compose:
    """Get a transform for tensors that are already float [0, 1].

    Args:
        img_size: Target image size (default 224).
    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


if __name__ == "__main__":
    import torch
    import numpy as np

    transform = get_default_transform()
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = transform(img)
    print(f"Input: numpy {img.shape}, Output: tensor {out.shape}")
    print(f"Output range: [{out.min():.2f}, {out.max():.2f}]")
