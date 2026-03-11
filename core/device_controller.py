"""
Android device controller using ADB.

Implements the Strategy Pattern with an abstract DeviceController protocol
and a concrete AndroidController implementation. Uses proper exceptions
instead of returning error strings.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from core.config import DeviceConfig
from core.logger import logger


class DeviceError(Exception):
    """Raised when a device command fails."""


@dataclass
class DeviceInfo:
    """Information about a connected device."""
    device_id: str
    width: int
    height: int


class DeviceController(ABC):
    """Abstract base for device controllers (Strategy Pattern)."""

    @abstractmethod
    def get_device_size(self) -> Tuple[int, int]:
        """Return (width, height) of the device screen."""

    @abstractmethod
    def get_screenshot(self, prefix: str, save_dir: Path) -> Path:
        """Capture a screenshot and save it locally."""

    @abstractmethod
    def get_xml(self, prefix: str, save_dir: Path) -> Path:
        """Dump UI hierarchy XML and save it locally."""

    @abstractmethod
    def tap(self, x: int, y: int) -> None:
        """Tap at the given screen coordinates."""

    @abstractmethod
    def text(self, input_str: str) -> None:
        """Input text on the device."""

    @abstractmethod
    def long_press(self, x: int, y: int, duration: int = 1000) -> None:
        """Long press at the given coordinates."""

    @abstractmethod
    def swipe(self, x: int, y: int, direction: str, dist: str = "medium") -> None:
        """Swipe from the given coordinates in a direction."""

    @abstractmethod
    def swipe_precise(self, start: Tuple[int, int], end: Tuple[int, int], duration: int = 400) -> None:
        """Swipe between two precise coordinate pairs."""

    @abstractmethod
    def back(self) -> None:
        """Press the back button."""


def _execute_adb(command: str, timeout: int = 10) -> str:
    """
    Execute an ADB command and return stdout.

    Raises:
        DeviceError: If the command fails or times out.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise DeviceError(
                f"ADB command failed: {command}\nstderr: {result.stderr.strip()}"
            )
        return result.stdout.strip()
    except subprocess.TimeoutExpired as e:
        raise DeviceError(f"ADB command timed out after {timeout}s: {command}") from e


def list_all_devices(timeout: int = 10) -> list[str]:
    """
    List all connected Android devices via ADB.

    Returns:
        List of device IDs.
    """
    result = _execute_adb("adb devices", timeout=timeout)
    devices = []
    for line in result.split("\n")[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


class AndroidController(DeviceController):
    """
    Concrete ADB-based Android device controller.

    Handles all device interactions: screenshots, XML dumps, tap, swipe,
    text input, long press, and back navigation.
    """

    def __init__(self, device_id: str, config: DeviceConfig) -> None:
        self.device_id = device_id
        self.screenshot_dir = config.screenshot_dir
        self.xml_dir = config.xml_dir
        self.timeout = config.adb_timeout
        self._width: int = 0
        self._height: int = 0
        self._width, self._height = self.get_device_size()
        logger.info(f"Connected to device {device_id} ({self._width}x{self._height})")

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def get_device_size(self) -> Tuple[int, int]:
        """Return (width, height) of the device screen."""
        result = _execute_adb(
            f"adb -s {self.device_id} shell wm size",
            timeout=self.timeout,
        )
        # Output format: "Physical size: 1080x2340"
        size_str = result.split(": ")[-1]
        w, h = size_str.split("x")
        return int(w), int(h)

    def get_screenshot(self, prefix: str, save_dir: Path) -> Path:
        """Capture a screenshot from the device and pull it locally."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        remote_path = f"{self.screenshot_dir}/{prefix}.png"
        local_path = save_dir / f"{prefix}.png"

        _execute_adb(
            f"adb -s {self.device_id} shell screencap -p {remote_path}",
            timeout=self.timeout,
        )
        _execute_adb(
            f"adb -s {self.device_id} pull {remote_path} {local_path}",
            timeout=self.timeout,
        )
        logger.debug(f"Screenshot saved: {local_path}")
        return local_path

    def get_xml(self, prefix: str, save_dir: Path) -> Path:
        """Dump UI hierarchy XML and pull it locally."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        remote_path = f"{self.xml_dir}/{prefix}.xml"
        local_path = save_dir / f"{prefix}.xml"

        _execute_adb(
            f"adb -s {self.device_id} shell uiautomator dump {remote_path}",
            timeout=self.timeout,
        )
        _execute_adb(
            f"adb -s {self.device_id} pull {remote_path} {local_path}",
            timeout=self.timeout,
        )
        logger.debug(f"XML hierarchy saved: {local_path}")
        return local_path

    def tap(self, x: int, y: int) -> None:
        """Tap at the given screen coordinates."""
        _execute_adb(
            f"adb -s {self.device_id} shell input tap {x} {y}",
            timeout=self.timeout,
        )
        logger.debug(f"Tapped at ({x}, {y})")

    def text(self, input_str: str) -> None:
        """Input text on the device."""
        # Escape special characters for ADB shell
        safe_str = input_str.replace(" ", "%s").replace("'", "").replace('"', '\\"')
        _execute_adb(
            f"adb -s {self.device_id} shell input text '{safe_str}'",
            timeout=self.timeout,
        )
        logger.debug(f"Entered text: {input_str[:50]}...")

    def long_press(self, x: int, y: int, duration: int = 1000) -> None:
        """Long press at the given coordinates."""
        _execute_adb(
            f"adb -s {self.device_id} shell input swipe {x} {y} {x} {y} {duration}",
            timeout=self.timeout,
        )
        logger.debug(f"Long pressed at ({x}, {y}) for {duration}ms")

    def swipe(self, x: int, y: int, direction: str, dist: str = "medium") -> None:
        """Swipe from the given coordinates in a direction."""
        unit_dist = self._width // 10
        dist_multiplier = {"short": 1, "medium": 2, "long": 3}
        unit_dist *= dist_multiplier.get(dist, 2)

        offset_map = {
            "up": (0, -2 * unit_dist),
            "down": (0, 2 * unit_dist),
            "left": (-unit_dist, 0),
            "right": (unit_dist, 0),
        }
        if direction not in offset_map:
            raise DeviceError(f"Invalid swipe direction: {direction}")

        dx, dy = offset_map[direction]
        _execute_adb(
            f"adb -s {self.device_id} shell input swipe {x} {y} {x + dx} {y + dy} 400",
            timeout=self.timeout,
        )
        logger.debug(f"Swiped {direction} ({dist}) from ({x}, {y})")

    def swipe_precise(
        self, start: Tuple[int, int], end: Tuple[int, int], duration: int = 400
    ) -> None:
        """Swipe between two precise coordinate pairs."""
        sx, sy = start
        ex, ey = end
        _execute_adb(
            f"adb -s {self.device_id} shell input swipe {sx} {sy} {ex} {ey} {duration}",
            timeout=self.timeout,
        )
        logger.debug(f"Precise swipe from ({sx}, {sy}) to ({ex}, {ey})")

    def back(self) -> None:
        """Press the back button."""
        _execute_adb(
            f"adb -s {self.device_id} shell input keyevent KEYCODE_BACK",
            timeout=self.timeout,
        )
        logger.debug("Pressed back button")
