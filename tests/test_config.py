"""Tests for the configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from core.config import load_config, reset_config, ConfigError, AppConfig


@pytest.fixture(autouse=True)
def clean_config():
    """Reset the singleton config before each test."""
    reset_config()
    yield
    reset_config()


def _write_config(path: Path, data: dict) -> None:
    """Helper to write a config YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _write_config(config_file, {
            "MODEL": "OpenAI",
            "OPENAI_API_KEY": "sk-test-key",
            "MAX_TOKENS": 512,
            "TEMPERATURE": 0.5,
            "VOICE_ENABLED": True,
            "VOICE_ENGINE": "whisper",
        })

        config = load_config(config_file)
        assert isinstance(config, AppConfig)
        assert config.model.provider == "OpenAI"
        assert config.model.openai_api_key == "sk-test-key"
        assert config.model.max_tokens == 512
        assert config.model.temperature == 0.5
        assert config.voice.enabled is True
        assert config.voice.engine == "whisper"

    def test_returns_singleton(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _write_config(config_file, {"MODEL": "OpenAI"})

        config1 = load_config(config_file)
        config2 = load_config(config_file)
        assert config1 is config2

    def test_raises_on_missing_file(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/config.yaml")

    def test_defaults_applied(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _write_config(config_file, {})

        config = load_config(config_file)
        assert config.agent.max_rounds == 20
        assert config.device.screenshot_dir == "/sdcard"
        assert config.voice.timeout == 10

    def test_env_var_override(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _write_config(config_file, {"MODEL": "OpenAI"})

        os.environ["VAA_MODEL"] = "Gemini"
        try:
            config = load_config(config_file)
            assert config.model.provider == "Gemini"
        finally:
            del os.environ["VAA_MODEL"]
