"""
UI hierarchy analyzer.

Parses Android UI hierarchy XML dumps to extract interactive elements
with their bounding boxes and attributes. Replaces the original project's
inline tree traversal with a clean UIAnalyzer class.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import List, Tuple

from core.logger import logger


@dataclass
class UIElement:
    """Represents an interactive UI element on the screen."""
    uid: str
    bbox: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1, y1), (x2, y2))
    attrib: str  # "clickable" or "focusable"
    text: str = ""
    content_desc: str = ""
    class_name: str = ""
    resource_id: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        """Return the center coordinates of the element."""
        (x1, y1), (x2, y2) = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def width(self) -> int:
        return self.bbox[1][0] - self.bbox[0][0]

    @property
    def height(self) -> int:
        return self.bbox[1][1] - self.bbox[0][1]


def _get_element_id(elem: ET.Element, parent: ET.Element | None = None) -> str:
    """
    Generate a unique ID for a UI element from its attributes.

    Combines resource-id, class name, dimensions, content-desc, and
    optionally the parent's ID for disambiguation.
    """
    bounds = elem.attrib.get("bounds", "[0,0][0,0]")[1:-1].split("][")
    x1, y1 = map(int, bounds[0].split(","))
    x2, y2 = map(int, bounds[1].split(","))
    elem_w, elem_h = x2 - x1, y2 - y1

    # Primary ID from resource-id or class+dimensions
    resource_id = elem.attrib.get("resource-id", "")
    if resource_id:
        elem_id = resource_id.replace(":", ".").replace("/", "_")
    else:
        class_name = elem.attrib.get("class", "Unknown")
        elem_id = f"{class_name}_{elem_w}_{elem_h}"

    # Append content-desc if short enough
    content_desc = elem.attrib.get("content-desc", "")
    if content_desc and len(content_desc) < 20:
        safe_desc = content_desc.replace("/", "_").replace(" ", "").replace(":", "_")
        elem_id += f"_{safe_desc}"

    # Prepend parent prefix for disambiguation
    if parent is not None:
        parent_id = _get_element_id(parent)
        elem_id = f"{parent_id}_{elem_id}"

    return elem_id


def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Euclidean distance between two points."""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class UIAnalyzer:
    """
    Analyzes Android UI hierarchy XML to extract interactive elements.

    Provides methods to parse the hierarchy, filter duplicates, and
    get a clean list of actionable elements.
    """

    def __init__(self, min_dist: int = 30) -> None:
        self.min_dist = min_dist

    def parse_hierarchy(
        self, xml_path: str | Path, attrib: str = "clickable"
    ) -> List[UIElement]:
        """
        Parse a UI hierarchy XML file and extract elements with the given attribute.

        Args:
            xml_path: Path to the UI hierarchy XML dump.
            attrib: Attribute to filter on ("clickable" or "focusable").

        Returns:
            List of UIElement objects.
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            logger.error(f"XML file not found: {xml_path}")
            return []

        elements: List[UIElement] = []
        path_stack: List[ET.Element] = []

        for event, elem in ET.iterparse(str(xml_path), ["start", "end"]):
            if event == "start":
                path_stack.append(elem)

                if elem.attrib.get(attrib) == "true":
                    parent = path_stack[-2] if len(path_stack) > 1 else None

                    bounds_str = elem.attrib.get("bounds", "[0,0][0,0]")
                    bounds = bounds_str[1:-1].split("][")
                    x1, y1 = map(int, bounds[0].split(","))
                    x2, y2 = map(int, bounds[1].split(","))
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Skip if too close to an existing element
                    is_close = any(
                        _distance(center, e.center) <= self.min_dist
                        for e in elements
                    )
                    if is_close:
                        continue

                    elem_id = _get_element_id(elem, parent)
                    index = elem.attrib.get("index", "0")
                    elem_id += f"_{index}"

                    elements.append(UIElement(
                        uid=elem_id,
                        bbox=((x1, y1), (x2, y2)),
                        attrib=attrib,
                        text=elem.attrib.get("text", ""),
                        content_desc=elem.attrib.get("content-desc", ""),
                        class_name=elem.attrib.get("class", ""),
                        resource_id=elem.attrib.get("resource-id", ""),
                    ))

            elif event == "end":
                if path_stack:
                    path_stack.pop()

        logger.debug(f"Parsed {len(elements)} {attrib} elements from {xml_path.name}")
        return elements

    def get_interactive_elements(self, xml_path: str | Path) -> List[UIElement]:
        """
        Get all interactive elements (clickable + focusable, deduplicated).

        This merges clickable and focusable elements, removing duplicates
        based on proximity.

        Args:
            xml_path: Path to the UI hierarchy XML dump.

        Returns:
            Combined, deduplicated list of UIElement objects.
        """
        clickable = self.parse_hierarchy(xml_path, "clickable")
        focusable = self.parse_hierarchy(xml_path, "focusable")

        # Start with clickable elements
        combined = list(clickable)

        # Add focusable elements that aren't too close to any clickable one
        for elem in focusable:
            is_close = any(
                _distance(elem.center, c.center) <= self.min_dist
                for c in clickable
            )
            if not is_close:
                combined.append(elem)

        logger.debug(
            f"Total interactive elements: {len(combined)} "
            f"({len(clickable)} clickable + {len(focusable)} focusable, deduplicated)"
        )
        return combined
