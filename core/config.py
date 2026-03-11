"""
Configuration loader with validation.

Uses the Singleton pattern to ensure a single configuration instance
throughout the application lifecycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


@dataclass
class VoiceConfig:
    """Voice input configuration."""
    enabled: bool = True
    engine: str = "whisper"           # "whisper" or "local"
    language: str = "en"
    timeout: int = 10
    energy_threshold: int = 300
    wake_word: str = "hey agent"


@dataclass
class ModelConfig:
    """LLM model configuration."""
    provider: str = "OpenAI"          # "OpenAI", "Gemini", "Qwen"
    # OpenAI
    openai_api_key: str = ""
    openai_api_model: str = "gpt-4o"
    openai_api_base: str = "https://api.openai.com/v1"
    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-pro"
    # Qwen
    dashscope_api_key: str = ""
    qwen_model: str = "qwen-vl-max"
    # Common
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class DeviceConfig:
    """Android device configuration."""
    screenshot_dir: str = "/sdcard"
    xml_dir: str = "/sdcard"
    adb_timeout: int = 10


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_rounds: int = 20
    request_interval: int = 3
    doc_refine: bool = False
    dark_mode: bool = False
    min_dist: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 2.0


@dataclass
class AppConfig:
    """Root configuration object."""
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


# Module-level singleton
_config: AppConfig | None = None


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """
    Load and validate configuration from a YAML file, merging with
    environment variable overrides.

    Returns the singleton AppConfig instance.
    """
    global _config
    if _config is not None:
        return _config

    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Merge environment variable overrides (VAA_ prefix)
    for key, value in os.environ.items():
        if key.startswith("VAA_"):
            config_key = key[4:]  # strip VAA_ prefix
            raw[config_key] = value

    _config = AppConfig(
        voice=VoiceConfig(
            enabled=raw.get("VOICE_ENABLED", True),
            engine=raw.get("VOICE_ENGINE", "whisper"),
            language=raw.get("VOICE_LANGUAGE", "en"),
            timeout=int(raw.get("VOICE_TIMEOUT", 10)),
            energy_threshold=int(raw.get("VOICE_ENERGY_THRESHOLD", 300)),
            wake_word=raw.get("WAKE_WORD", "hey agent"),
        ),
        model=ModelConfig(
            provider=raw.get("MODEL", "OpenAI"),
            openai_api_key=raw.get("OPENAI_API_KEY", ""),
            openai_api_model=raw.get("OPENAI_API_MODEL", "gpt-4o"),
            openai_api_base=raw.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            gemini_api_key=raw.get("GEMINI_API_KEY", ""),
            gemini_model=raw.get("GEMINI_MODEL", "gemini-1.5-pro"),
            dashscope_api_key=raw.get("DASHSCOPE_API_KEY", ""),
            qwen_model=raw.get("QWEN_MODEL", "qwen-vl-max"),
            max_tokens=int(raw.get("MAX_TOKENS", 1024)),
            temperature=float(raw.get("TEMPERATURE", 0.0)),
        ),
        device=DeviceConfig(
            screenshot_dir=raw.get("ANDROID_SCREENSHOT_DIR", "/sdcard"),
            xml_dir=raw.get("ANDROID_XML_DIR", "/sdcard"),
            adb_timeout=int(raw.get("ADB_TIMEOUT", 10)),
        ),
        agent=AgentConfig(
            max_rounds=int(raw.get("MAX_ROUNDS", 20)),
            request_interval=int(raw.get("REQUEST_INTERVAL", 3)),
            doc_refine=bool(raw.get("DOC_REFINE", False)),
            dark_mode=bool(raw.get("DARK_MODE", False)),
            min_dist=int(raw.get("MIN_DIST", 30)),
            max_retries=int(raw.get("MAX_RETRIES", 3)),
            retry_backoff_factor=float(raw.get("RETRY_BACKOFF_FACTOR", 2.0)),
        ),
    )
    return _config


def reset_config() -> None:
    """Reset the singleton config (useful for tests)."""
    global _config
    _config = None
