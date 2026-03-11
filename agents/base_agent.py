"""
Abstract base agent defining the agent lifecycle.

Uses the Template Method Pattern to define a common setup/run/teardown
lifecycle for all agent types (explorer, task runner, demo recorder).
"""

from __future__ import annotations

import datetime
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from core.config import AppConfig, load_config
from core.device_controller import (
    AndroidController,
    DeviceController,
    DeviceError,
    list_all_devices,
)
from core.logger import logger, print_colored
from core.models import LLMModel, ModelFactory
from core.voice_input import VoiceInputManager


class AgentError(Exception):
    """Raised when an agent operation fails."""


class BaseAgent(ABC):
    """
    Abstract base agent implementing the Template Method Pattern.

    Subclasses implement _execute() with their specific logic.
    The base class handles device connection, model initialization,
    directory setup, and input gathering via voice or keyboard.
    """

    def __init__(
        self,
        app_name: str,
        root_dir: str | Path = "./",
        voice_enabled: bool = False,
        config_path: str | Path = "config.yaml",
    ) -> None:
        self.app_name = app_name.replace(" ", "")
        self.root_dir = Path(root_dir)
        self.config_path = Path(config_path)

        # Loaded during setup
        self.config: AppConfig = None  # type: ignore
        self.controller: DeviceController = None  # type: ignore
        self.model: LLMModel = None  # type: ignore
        self.voice: VoiceInputManager = None  # type: ignore
        self._voice_override = voice_enabled

    def run(self) -> None:
        """
        Execute the full agent lifecycle: setup → execute → teardown.

        This is the Template Method that orchestrates the agent.
        """
        try:
            self._setup()
            self._execute()
        except KeyboardInterrupt:
            logger.warning("Agent interrupted by user")
        except AgentError as e:
            logger.error(f"Agent error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self._teardown()

    def _setup(self) -> None:
        """Initialize config, device, model, and voice input."""
        logger.info(f"Setting up agent for app: {self.app_name}")

        # Load configuration
        self.config = load_config(self.config_path)

        # Override voice setting from CLI
        if self._voice_override:
            self.config.voice.enabled = True

        # Initialize voice input
        self.voice = VoiceInputManager(self.config.voice)

        # Initialize LLM model
        self.model = ModelFactory.create(self.config.model)

        # Connect to device
        self.controller = self._connect_device()

        logger.info("Agent setup complete")

    def _connect_device(self) -> DeviceController:
        """Discover and connect to an Android device."""
        device_list = list_all_devices(timeout=self.config.device.adb_timeout)
        if not device_list:
            raise AgentError("No Android devices found. Is ADB running?")

        logger.info(f"Devices found: {device_list}")

        if len(device_list) == 1:
            device_id = device_list[0]
            logger.info(f"Auto-selected device: {device_id}")
        else:
            device_id = self.voice.get_input(
                f"Multiple devices found. Choose one:\n"
                + "\n".join(f"  {i+1}. {d}" for i, d in enumerate(device_list))
                + "\nEnter device ID:"
            )

        return AndroidController(device_id, self.config.device)

    def _get_task_description(self) -> str:
        """Get the task description from the user (voice or keyboard)."""
        return self.voice.get_input(
            "Please describe the task you want me to complete:"
        )

    def _create_work_dirs(self, *dirs: Path) -> None:
        """Create working directories."""
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _generate_timestamp_name(self, prefix: str) -> str:
        """Generate a timestamped directory name."""
        ts = int(time.time())
        return datetime.datetime.fromtimestamp(ts).strftime(
            f"{prefix}_{self.app_name}_%Y-%m-%d_%H-%M-%S"
        )

    @abstractmethod
    def _execute(self) -> None:
        """
        Execute the agent's main logic.

        Subclasses must implement this with their specific behavior
        (exploration, task execution, demo recording, etc.).
        """

    def _teardown(self) -> None:
        """Cleanup after agent execution."""
        logger.info("Agent teardown complete")
