"""Trajectory dataset for offline (obs, action) pairs."""

from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset for offline trajectories with sub-trajectory sampling.

    Loads offline (observation, action) pairs and samples sub-trajectories
    of a given length with frame skipping.
    """

    def __init__(
        self,
        trajectories: list[dict],
        sub_traj_length: int = 4,
        frame_skip: int = 5,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            trajectories: List of dicts, each with:
                'observations': np.ndarray (L, H, W, 3) uint8 images
                'actions': np.ndarray (L, A) float32 actions
            sub_traj_length: T, number of frames in each sub-trajectory.
            frame_skip: Number of raw frames to skip between samples.
            transform: Optional torchvision transform for images.
        """
        self.trajectories = trajectories
        self.T = sub_traj_length
        self.skip = frame_skip
        self.transform = transform

        # Pre-compute valid (trajectory_idx, start_frame) pairs
        self.valid_indices: list[tuple[int, int]] = []
        required_length = (sub_traj_length - 1) * frame_skip + 1
        for traj_idx, traj in enumerate(trajectories):
            traj_len = len(traj["observations"])
            for start in range(traj_len - required_length + 1):
                self.valid_indices.append((traj_idx, start))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx, start = self.valid_indices[idx]
        traj = self.trajectories[traj_idx]

        # Extract frames with skip
        frame_indices = [start + t * self.skip for t in range(self.T)]
        obs_list = [traj["observations"][i] for i in frame_indices]
        actions_list = [traj["actions"][i] for i in frame_indices]

        # Apply transforms to each observation
        if self.transform:
            obs_list = [self.transform(o) for o in obs_list]
        else:
            # Default: convert numpy HWC uint8 to CHW float tensor
            obs_list = [
                torch.from_numpy(o).permute(2, 0, 1).float() / 255.0
                if isinstance(o, np.ndarray)
                else o
                for o in obs_list
            ]

        obs = torch.stack(obs_list)  # (T, 3, H, W)
        actions = torch.tensor(
            np.array(actions_list), dtype=torch.float32
        )  # (T, A)

        return {"obs": obs, "actions": actions}


def make_synthetic_dataset(
    num_trajectories: int = 10,
    traj_length: int = 50,
    img_size: int = 64,
    action_dim: int = 2,
) -> list[dict]:
    """Generate random trajectories for testing.

    Args:
        num_trajectories: Number of trajectories to generate.
        traj_length: Length of each trajectory.
        img_size: Height and width of synthetic images.
        action_dim: Dimension of action vectors.
    Returns:
        List of trajectory dicts with 'observations' and 'actions'.
    """
    trajectories = []
    for _ in range(num_trajectories):
        trajectories.append({
            "observations": np.random.randint(
                0, 255, (traj_length, img_size, img_size, 3), dtype=np.uint8
            ),
            "actions": np.random.randn(traj_length, action_dim).astype(
                np.float32
            ),
        })
    return trajectories


if __name__ == "__main__":
    from lewm.data.transforms import get_default_transform

    trajs = make_synthetic_dataset(num_trajectories=5, traj_length=30)
    transform = get_default_transform(img_size=224)
    dataset = TrajectoryDataset(
        trajs, sub_traj_length=4, frame_skip=5, transform=transform
    )
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Obs shape: {sample['obs'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch obs: {batch['obs'].shape}")
    print(f"Batch actions: {batch['actions'].shape}")
