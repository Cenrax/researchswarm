"""Configuration models for the CAID framework.

Provides Pydantic-based configuration with YAML loading and
environment variable overrides. Default values match the paper's
recommended settings (Section 3).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM provider and model."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20260120"
    api_key: str = Field(
        default_factory=lambda: os.environ.get("CAID_LLM_API_KEY", "")
    )
    temperature: float = 0.0
    max_tokens: int = 4096


class CAIDConfig(BaseModel):
    """Top-level CAID framework configuration.

    Attributes:
        mode: Operating mode - 'commit0' for repo stub filling or
              'paperbench' for paper reproduction.
        repo_path: Path to the target git repository.
        task_description: Free-text or path to paper description.
        max_engineers: Number of concurrent engineer agents.
        manager_max_iterations: Manager loop iteration cap.
        engineer_max_iterations: Per-engineer iteration cap.
        implementation_rounds: Number of full delegation rounds.
        restricted_files: Files engineers must not modify.
        llm: LLM provider configuration.
        worktree_base_dir: Directory for git worktrees.
    """

    mode: Literal["commit0", "paperbench"] = "commit0"
    repo_path: Path = Path(".")
    task_description: str = ""
    max_engineers: int = 4
    manager_max_iterations: int = 50
    engineer_max_iterations: int = 80
    implementation_rounds: int = 2
    restricted_files: list[str] = Field(
        default_factory=lambda: ["__init__.py"]
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)
    worktree_base_dir: Path = Path("/tmp/caid-worktrees")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CAIDConfig:
        """Load configuration from a YAML file.

        Environment variables override YAML values when set:
          - CAID_LLM_API_KEY overrides llm.api_key
          - CAID_MODE overrides mode
          - CAID_MAX_ENGINEERS overrides max_engineers
        """
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides: dict[str, str] = {
            "CAID_MODE": "mode",
            "CAID_MAX_ENGINEERS": "max_engineers",
            "CAID_REPO_PATH": "repo_path",
        }
        for env_var, config_key in env_overrides.items():
            val = os.environ.get(env_var)
            if val is not None:
                if config_key == "max_engineers":
                    data[config_key] = int(val)
                else:
                    data[config_key] = val

        # Handle nested LLM config env vars
        llm_data = data.get("llm", {})
        if isinstance(llm_data, dict):
            api_key = os.environ.get("CAID_LLM_API_KEY")
            if api_key:
                llm_data["api_key"] = api_key
            data["llm"] = llm_data

        return cls(**data)


if __name__ == "__main__":
    # Quick verification
    cfg = CAIDConfig()
    print(f"Default config: mode={cfg.mode}, engineers={cfg.max_engineers}")
    print(f"Manager iterations: {cfg.manager_max_iterations}")
    print(f"Engineer iterations: {cfg.engineer_max_iterations}")
    print(f"LLM model: {cfg.llm.model}")
    print(f"Worktree dir: {cfg.worktree_base_dir}")

    # Test YAML round-trip
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg.model_dump(mode="json"), f)
        tmp_path = f.name
    loaded = CAIDConfig.from_yaml(tmp_path)
    print(f"YAML round-trip: mode={loaded.mode}, engineers={loaded.max_engineers}")
    os.unlink(tmp_path)
    print("Config module OK")
