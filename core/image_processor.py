"""
Image processing for screenshots.

Provides screenshot labeling (numbering interactive elements) and grid
overlay drawing using OpenCV. Replaces the original project's utils.py
drawing functions with a cleaner ImageProcessor class.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from core.logger import logger
from core.ui_analyzer import UIElement


class ImageProcessor:
    """Handles screenshot labeling and grid overlay operations."""

    @staticmethod
    def label_elements(
        image_path: str | Path,
        output_path: str | Path,
        elements: List[UIElement],
        record_mode: bool = False,
        dark_mode: bool = False,
    ) -> np.ndarray:
        """
        Draw numbered labels on interactive elements in a screenshot.

        Args:
            image_path: Path to the source screenshot.
            output_path: Path to save the labeled image.
            elements: List of UI elements to label.
            record_mode: If True, color-code by attribute type.
            dark_mode: Adjust label colors for dark backgrounds.

        Returns:
            The labeled image as a numpy array.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        for idx, elem in enumerate(elements, start=1):
            (x1, y1), (x2, y2) = elem.bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = str(idx)

            # Color selection
            if record_mode:
                color_map = {
                    "clickable": (250, 0, 0),     # Blue (BGR)
                    "focusable": (0, 0, 250),      # Red (BGR)
                }
                bg_color = color_map.get(elem.attrib, (0, 250, 0))
                text_color = (255, 255, 255)
            else:
                if dark_mode:
                    bg_color = (255, 250, 250)
                    text_color = (10, 10, 10)
                else:
                    bg_color = (10, 10, 10)
                    text_color = (255, 250, 250)

            # Draw label background + text
            try:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Position label near center, offset to avoid overlap
                lx = cx + 5
                ly = cy + 5

                # Background rectangle
                cv2.rectangle(
                    img,
                    (lx - 2, ly - th - 4),
                    (lx + tw + 4, ly + 4),
                    bg_color,
                    cv2.FILLED,
                )
                # Label text
                cv2.putText(
                    img, label, (lx, ly),
                    font, font_scale, text_color, thickness, cv2.LINE_AA,
                )
            except Exception as e:
                logger.warning(f"Failed to label element {idx}: {e}")

        cv2.imwrite(str(output_path), img)
        logger.debug(f"Labeled screenshot saved: {output_path}")
        return img

    @staticmethod
    def draw_grid(
        image_path: str | Path,
        output_path: str | Path,
    ) -> Tuple[int, int]:
        """
        Draw a numbered grid overlay on a screenshot.

        Args:
            image_path: Path to the source screenshot.
            output_path: Path to save the grid image.

        Returns:
            Tuple of (rows, cols) in the grid.
        """
        def _find_unit_length(n: int) -> int:
            """Find a divisor of n between 120 and 180."""
            for i in range(1, n + 1):
                if n % i == 0 and 120 <= i <= 180:
                    return i
            return 120  # Fallback

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        height, width = img.shape[:2]
        color = (255, 116, 113)  # Light coral color

        unit_h = _find_unit_length(height)
        unit_w = _find_unit_length(width)
        thickness = max(1, unit_w // 50)

        rows = height // unit_h
        cols = width // unit_w

        for i in range(rows):
            for j in range(cols):
                label = i * cols + j + 1
                x1 = j * unit_w
                y1 = i * unit_h
                x2 = (j + 1) * unit_w
                y2 = (i + 1) * unit_h

                # Grid cell rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness // 2))

                # Label with shadow for readability
                label_x = x1 + int(unit_w * 0.05)
                label_y = y1 + int(unit_h * 0.3)
                font_scale = int(0.01 * unit_w)

                # Shadow
                cv2.putText(
                    img, str(label),
                    (label_x + 3, label_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA,
                )
                # Foreground
                cv2.putText(
                    img, str(label),
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness, cv2.LINE_AA,
                )

        cv2.imwrite(str(output_path), img)
        logger.debug(f"Grid overlay saved: {output_path} ({rows}x{cols})")
        return rows, cols

    @staticmethod
    def encode_base64(image_path: str | Path) -> str:
        """Encode an image file to base64 string."""
        with open(str(image_path), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
