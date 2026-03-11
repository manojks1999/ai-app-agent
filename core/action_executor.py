"""
Action parsing and execution using the Command Pattern.

Parses LLM text responses into structured Action commands, then executes
them on the device controller. Each action is a self-contained Command
object with all parameters needed for execution.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.device_controller import DeviceController, DeviceError
from core.logger import logger
from core.ui_analyzer import UIElement


class ActionError(Exception):
    """Raised when action parsing or execution fails."""


# ─── Action Data Classes ─────────────────────────────────────────────────────

@dataclass
class ParsedAction:
    """Base class for parsed actions."""
    name: str
    summary: str = ""


@dataclass
class TapAction(ParsedAction):
    """Tap an element by its label number."""
    element_index: int = 0


@dataclass
class TextAction(ParsedAction):
    """Type text into a field."""
    text: str = ""


@dataclass
class LongPressAction(ParsedAction):
    """Long press an element by its label number."""
    element_index: int = 0


@dataclass
class SwipeAction(ParsedAction):
    """Swipe an element in a direction."""
    element_index: int = 0
    direction: str = ""
    distance: str = "medium"


@dataclass
class GridAction(ParsedAction):
    """Activate grid overlay mode."""
    pass


@dataclass
class TapGridAction(ParsedAction):
    """Tap a specific grid area + subarea."""
    area: int = 0
    subarea: str = "center"


@dataclass
class LongPressGridAction(ParsedAction):
    """Long press a specific grid area + subarea."""
    area: int = 0
    subarea: str = "center"


@dataclass
class SwipeGridAction(ParsedAction):
    """Swipe between two grid areas."""
    start_area: int = 0
    start_subarea: str = "center"
    end_area: int = 0
    end_subarea: str = "center"


@dataclass
class FinishAction(ParsedAction):
    """Task is complete."""
    pass


@dataclass
class ErrorAction(ParsedAction):
    """Parsing error occurred."""
    error_message: str = ""


# ─── Reflection Decision ─────────────────────────────────────────────────────

@dataclass
class ReflectionResult:
    """Result from the reflection/self-evaluation step."""
    decision: str  # "BACK", "CONTINUE", "INEFFECTIVE", "SUCCESS", "ERROR"
    thought: str = ""
    documentation: str = ""


# ─── Action Parser ────────────────────────────────────────────────────────────

class ActionParser:
    """
    Parses LLM text responses into structured action objects.

    Supports two modes:
    - Element mode: actions reference labeled UI elements
    - Grid mode: actions reference grid area + subarea
    """

    @staticmethod
    def parse_explore_response(response: str) -> ParsedAction:
        """
        Parse an exploration/task response into an action.

        Expected format:
            Observation: <text>
            Thought: <text>
            Action: <function call or FINISH>
            Summary: <text>
        """
        try:
            observation = re.findall(r"Observation: (.*?)$", response, re.MULTILINE)
            thought = re.findall(r"Thought: (.*?)$", response, re.MULTILINE)
            action = re.findall(r"Action: (.*?)$", response, re.MULTILINE)
            summary = re.findall(r"Summary: (.*?)$", response, re.MULTILINE)

            obs_text = observation[0] if observation else ""
            think_text = thought[0] if thought else ""
            act_text = action[0] if action else ""
            sum_text = summary[0] if summary else ""

            logger.info(f"Observation: {obs_text}")
            logger.info(f"Thought: {think_text}")
            logger.info(f"Action: {act_text}")
            logger.info(f"Summary: {sum_text}")

            if "FINISH" in act_text:
                return FinishAction(name="FINISH", summary=sum_text)

            act_name = act_text.split("(")[0].strip()

            if act_name == "tap":
                area = int(re.findall(r"tap\((.*?)\)", act_text)[0])
                return TapAction(name="tap", element_index=area, summary=sum_text)

            elif act_name == "text":
                input_str = re.findall(r'text\((.*?)\)', act_text)[0]
                # Remove surrounding quotes
                input_str = input_str.strip('"').strip("'")
                return TextAction(name="text", text=input_str, summary=sum_text)

            elif act_name == "long_press":
                area = int(re.findall(r"long_press\((.*?)\)", act_text)[0])
                return LongPressAction(
                    name="long_press", element_index=area, summary=sum_text
                )

            elif act_name == "swipe":
                params = re.findall(r"swipe\((.*?)\)", act_text)[0]
                parts = [p.strip() for p in params.split(",")]
                area = int(parts[0])
                direction = parts[1].strip('"').strip("'")
                dist = parts[2].strip('"').strip("'") if len(parts) > 2 else "medium"
                return SwipeAction(
                    name="swipe",
                    element_index=area,
                    direction=direction,
                    distance=dist,
                    summary=sum_text,
                )

            elif act_name == "grid":
                return GridAction(name="grid", summary=sum_text)

            else:
                return ErrorAction(
                    name="ERROR",
                    error_message=f"Undefined action: {act_name}",
                    summary=sum_text,
                )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response}")
            return ErrorAction(
                name="ERROR", error_message=str(e), summary=""
            )

    @staticmethod
    def parse_grid_response(response: str) -> ParsedAction:
        """Parse a grid-mode response into an action."""
        try:
            observation = re.findall(r"Observation: (.*?)$", response, re.MULTILINE)
            thought = re.findall(r"Thought: (.*?)$", response, re.MULTILINE)
            action = re.findall(r"Action: (.*?)$", response, re.MULTILINE)
            summary = re.findall(r"Summary: (.*?)$", response, re.MULTILINE)

            act_text = action[0] if action else ""
            sum_text = summary[0] if summary else ""

            if "FINISH" in act_text:
                return FinishAction(name="FINISH", summary=sum_text)

            act_name = act_text.split("(")[0].strip()

            if act_name == "tap":
                params = re.findall(r"tap\((.*?)\)", act_text)[0].split(",")
                area = int(params[0].strip())
                subarea = params[1].strip().strip('"').strip("'")
                return TapGridAction(
                    name="tap_grid", area=area, subarea=subarea, summary=sum_text
                )

            elif act_name == "long_press":
                params = re.findall(r"long_press\((.*?)\)", act_text)[0].split(",")
                area = int(params[0].strip())
                subarea = params[1].strip().strip('"').strip("'")
                return LongPressGridAction(
                    name="long_press_grid", area=area, subarea=subarea, summary=sum_text
                )

            elif act_name == "swipe":
                params = re.findall(r"swipe\((.*?)\)", act_text)[0].split(",")
                return SwipeGridAction(
                    name="swipe_grid",
                    start_area=int(params[0].strip()),
                    start_subarea=params[1].strip().strip('"').strip("'"),
                    end_area=int(params[2].strip()),
                    end_subarea=params[3].strip().strip('"').strip("'"),
                    summary=sum_text,
                )

            elif act_name == "grid":
                return GridAction(name="grid", summary=sum_text)

            else:
                return ErrorAction(
                    name="ERROR",
                    error_message=f"Undefined grid action: {act_name}",
                    summary=sum_text,
                )

        except Exception as e:
            logger.error(f"Failed to parse grid response: {e}")
            return ErrorAction(name="ERROR", error_message=str(e))

    @staticmethod
    def parse_reflection(response: str) -> ReflectionResult:
        """
        Parse a self-reflection/evaluation response.

        Expected format:
            Decision: <BACK|CONTINUE|INEFFECTIVE|SUCCESS>
            Thought: <text>
            Documentation: <text> (for BACK, CONTINUE, SUCCESS)
        """
        try:
            decision = re.findall(r"Decision: (.*?)$", response, re.MULTILINE)
            thought = re.findall(r"Thought: (.*?)$", response, re.MULTILINE)

            dec_text = decision[0].strip() if decision else ""
            think_text = thought[0].strip() if thought else ""

            logger.info(f"Reflection Decision: {dec_text}")
            logger.info(f"Reflection Thought: {think_text}")

            if dec_text == "INEFFECTIVE":
                return ReflectionResult(decision=dec_text, thought=think_text)

            if dec_text in ("BACK", "CONTINUE", "SUCCESS"):
                doc = re.findall(r"Documentation: (.*?)$", response, re.MULTILINE)
                doc_text = doc[0].strip() if doc else ""
                logger.info(f"Documentation: {doc_text}")
                return ReflectionResult(
                    decision=dec_text, thought=think_text, documentation=doc_text
                )

            return ReflectionResult(
                decision="ERROR", thought=f"Undefined decision: {dec_text}"
            )

        except Exception as e:
            logger.error(f"Failed to parse reflection: {e}")
            return ReflectionResult(decision="ERROR", thought=str(e))


# ─── Action Executor ──────────────────────────────────────────────────────────

class ActionExecutor:
    """
    Executes parsed actions on a device controller.

    Handles coordinate resolution from element indices and grid areas.
    """

    def __init__(
        self,
        controller: DeviceController,
        screen_width: int,
        screen_height: int,
    ) -> None:
        self.controller = controller
        self.screen_width = screen_width
        self.screen_height = screen_height

    def execute(
        self,
        action: ParsedAction,
        elements: List[UIElement] | None = None,
        grid_rows: int = 0,
        grid_cols: int = 0,
    ) -> bool:
        """
        Execute a parsed action on the device.

        Args:
            action: The parsed action to execute.
            elements: UI elements list (for element-mode actions).
            grid_rows: Number of grid rows (for grid-mode actions).
            grid_cols: Number of grid cols (for grid-mode actions).

        Returns:
            True if the action was executed successfully.

        Raises:
            ActionError: If execution fails.
        """
        try:
            if isinstance(action, FinishAction):
                logger.info("Task marked as FINISH")
                return True

            elif isinstance(action, TapAction):
                if not elements:
                    raise ActionError("No elements list for tap action")
                tl, br = elements[action.element_index - 1].bbox
                x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                self.controller.tap(x, y)

            elif isinstance(action, TextAction):
                self.controller.text(action.text)

            elif isinstance(action, LongPressAction):
                if not elements:
                    raise ActionError("No elements list for long_press action")
                tl, br = elements[action.element_index - 1].bbox
                x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                self.controller.long_press(x, y)

            elif isinstance(action, SwipeAction):
                if not elements:
                    raise ActionError("No elements list for swipe action")
                tl, br = elements[action.element_index - 1].bbox
                x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                self.controller.swipe(x, y, action.direction, action.distance)

            elif isinstance(action, GridAction):
                logger.info("Grid mode activated")
                return True

            elif isinstance(action, TapGridAction):
                x, y = self._area_to_xy(
                    action.area, action.subarea, grid_rows, grid_cols
                )
                self.controller.tap(x, y)

            elif isinstance(action, LongPressGridAction):
                x, y = self._area_to_xy(
                    action.area, action.subarea, grid_rows, grid_cols
                )
                self.controller.long_press(x, y)

            elif isinstance(action, SwipeGridAction):
                sx, sy = self._area_to_xy(
                    action.start_area, action.start_subarea, grid_rows, grid_cols
                )
                ex, ey = self._area_to_xy(
                    action.end_area, action.end_subarea, grid_rows, grid_cols
                )
                self.controller.swipe_precise((sx, sy), (ex, ey))

            elif isinstance(action, ErrorAction):
                raise ActionError(f"Error action: {action.error_message}")

            else:
                raise ActionError(f"Unknown action type: {type(action).__name__}")

            return True

        except DeviceError as e:
            raise ActionError(f"Device error during {action.name}: {e}") from e
        except IndexError as e:
            raise ActionError(
                f"Invalid element index in {action.name}: {e}"
            ) from e

    def _area_to_xy(
        self,
        area: int,
        subarea: str,
        rows: int,
        cols: int,
    ) -> Tuple[int, int]:
        """Convert grid area + subarea to screen coordinates."""
        if rows == 0 or cols == 0:
            raise ActionError("Grid dimensions not set")

        area_idx = area - 1
        row, col = area_idx // cols, area_idx % cols
        cell_w = self.screen_width // cols
        cell_h = self.screen_height // rows
        x0 = col * cell_w
        y0 = row * cell_h

        subarea_offsets = {
            "top-left": (cell_w // 4, cell_h // 4),
            "top": (cell_w // 2, cell_h // 4),
            "top-right": (cell_w * 3 // 4, cell_h // 4),
            "left": (cell_w // 4, cell_h // 2),
            "center": (cell_w // 2, cell_h // 2),
            "right": (cell_w * 3 // 4, cell_h // 2),
            "bottom-left": (cell_w // 4, cell_h * 3 // 4),
            "bottom": (cell_w // 2, cell_h * 3 // 4),
            "bottom-right": (cell_w * 3 // 4, cell_h * 3 // 4),
        }

        dx, dy = subarea_offsets.get(subarea, (cell_w // 2, cell_h // 2))
        return x0 + dx, y0 + dy
