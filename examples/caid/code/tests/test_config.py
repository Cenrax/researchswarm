"""Tests for caid.config module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from caid.config import CAIDConfig, LLMConfig


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-sonnet-4-5-20260120"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096

    def test_custom_values(self) -> None:
        cfg = LLMConfig(model="gpt-4", provider="openai", api_key="sk-test")
        assert cfg.model == "gpt-4"
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test"


class TestCAIDConfig:
    def test_defaults_match_paper(self, sample_config: CAIDConfig) -> None:
        """Verify defaults match CAID paper values."""
        assert sample_config.mode == "commit0"
        assert sample_config.max_engineers == 4
        assert sample_config.manager_max_iterations == 50
        assert sample_config.engineer_max_iterations == 80
        assert sample_config.implementation_rounds == 2
        assert "__init__.py" in sample_config.restricted_files

    def test_paperbench_mode(self) -> None:
        cfg = CAIDConfig(mode="paperbench", max_engineers=2)
        assert cfg.mode == "paperbench"
        assert cfg.max_engineers == 2

    def test_from_yaml(self) -> None:
        data = {
            "mode": "paperbench",
            "max_engineers": 2,
            "llm": {"model": "gpt-4", "api_key": "test"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = CAIDConfig.from_yaml(path)
            assert cfg.mode == "paperbench"
            assert cfg.max_engineers == 2
            assert cfg.llm.model == "gpt-4"
        finally:
            os.unlink(path)

    def test_from_yaml_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            cfg = CAIDConfig.from_yaml(path)
            assert cfg.mode == "commit0"  # defaults
        finally:
            os.unlink(path)

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"mode": "commit0", "max_engineers": 4}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            monkeypatch.setenv("CAID_MODE", "paperbench")
            monkeypatch.setenv("CAID_MAX_ENGINEERS", "2")
            cfg = CAIDConfig.from_yaml(path)
            assert cfg.mode == "paperbench"
            assert cfg.max_engineers == 2
        finally:
            os.unlink(path)

    def test_api_key_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"llm": {"model": "gpt-4"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            monkeypatch.setenv("CAID_LLM_API_KEY", "sk-from-env")
            cfg = CAIDConfig.from_yaml(path)
            assert cfg.llm.api_key == "sk-from-env"
        finally:
            os.unlink(path)

    def test_yaml_round_trip(self) -> None:
        original = CAIDConfig(mode="commit0", max_engineers=3)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(original.model_dump(mode="json"), f)
            path = f.name
        try:
            loaded = CAIDConfig.from_yaml(path)
            assert loaded.mode == original.mode
            assert loaded.max_engineers == original.max_engineers
        finally:
            os.unlink(path)
