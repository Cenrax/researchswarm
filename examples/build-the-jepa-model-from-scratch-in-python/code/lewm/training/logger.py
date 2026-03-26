"""Simple logging utilities for training."""

import time
from typing import Optional


class TrainingLogger:
    """Simple logger that prints training statistics."""

    def __init__(self, log_interval: int = 1) -> None:
        """Initialize logger.

        Args:
            log_interval: Print every N epochs.
        """
        self.log_interval = log_interval
        self.history: list[dict] = []
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Mark the start of training."""
        self.start_time = time.time()

    def log_epoch(self, epoch: int, num_epochs: int, stats: dict) -> None:
        """Log statistics for one epoch.

        Args:
            epoch: Current epoch number (1-based).
            num_epochs: Total number of epochs.
            stats: Dictionary of metric names to values.
        """
        self.history.append({"epoch": epoch, **stats})

        if epoch % self.log_interval == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            parts = [f"Epoch {epoch}/{num_epochs}"]
            for key, value in stats.items():
                if isinstance(value, float):
                    parts.append(f"{key}: {value:.6f}")
                else:
                    parts.append(f"{key}: {value}")
            parts.append(f"[{elapsed:.1f}s]")
            print(" | ".join(parts))

    def log_warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Warning text to display.
        """
        print(f"WARNING: {message}")

    def get_history(self) -> list[dict]:
        """Return the full training history."""
        return self.history


if __name__ == "__main__":
    logger = TrainingLogger(log_interval=1)
    logger.start()
    for epoch in range(1, 4):
        stats = {"pred_loss": 1.0 / epoch, "sigreg_loss": 0.1 / epoch}
        logger.log_epoch(epoch, 3, stats)
    print(f"History length: {len(logger.get_history())}")
