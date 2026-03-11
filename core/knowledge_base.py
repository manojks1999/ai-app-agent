"""
Knowledge base for UI element documentation.

Uses the Repository Pattern to manage documentation storage and retrieval.
Stores docs as JSON files instead of the original project's unsafe
ast.literal_eval approach. Provides clean CRUD operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from core.logger import logger


@dataclass
class ElementDoc:
    """Documentation for a single UI element, organized by action type."""
    tap: str = ""
    text: str = ""
    long_press: str = ""
    v_swipe: str = ""
    h_swipe: str = ""


class KnowledgeBase:
    """
    Repository for UI element documentation.

    Stores documentation as JSON files, one per element, organized by
    action type. Supports creation, updating, and retrieval.
    """

    def __init__(self, docs_dir: str | Path) -> None:
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def _doc_path(self, element_id: str) -> Path:
        """Get the file path for an element's documentation."""
        safe_id = element_id.replace("/", "_").replace(":", ".")
        return self.docs_dir / f"{safe_id}.json"

    def get_doc(self, element_id: str) -> Optional[ElementDoc]:
        """
        Retrieve documentation for a specific element.

        Args:
            element_id: Unique identifier of the UI element.

        Returns:
            ElementDoc if found, None otherwise.
        """
        doc_path = self._doc_path(element_id)
        if not doc_path.exists():
            return None

        try:
            with open(doc_path, "r") as f:
                data = json.load(f)
            return ElementDoc(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load doc for {element_id}: {e}")
            return None

    def save_doc(
        self,
        element_id: str,
        action_type: str,
        doc_text: str,
    ) -> None:
        """
        Save or update documentation for an element's action type.

        Args:
            element_id: Unique identifier of the UI element.
            action_type: The action type (tap, text, long_press, v_swipe, h_swipe).
            doc_text: The documentation text to save.
        """
        existing = self.get_doc(element_id) or ElementDoc()

        if hasattr(existing, action_type):
            setattr(existing, action_type, doc_text)
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return

        doc_path = self._doc_path(element_id)
        with open(doc_path, "w") as f:
            json.dump(asdict(existing), f, indent=2)

        logger.info(f"Documentation saved: {element_id} ({action_type})")

    def has_doc(self, element_id: str, action_type: str = "") -> bool:
        """
        Check if documentation exists for an element.

        Args:
            element_id: Unique identifier of the UI element.
            action_type: Optional — if provided, checks only this action type.

        Returns:
            True if documentation exists.
        """
        doc = self.get_doc(element_id)
        if doc is None:
            return False
        if action_type:
            return bool(getattr(doc, action_type, ""))
        return True

    def get_docs_for_elements(
        self,
        element_ids: List[str],
    ) -> str:
        """
        Generate a formatted documentation string for multiple elements.

        This is used to inject UI documentation into the task prompt.

        Args:
            element_ids: List of (index, element_id) tuples, where index
                          is the 1-based label number.

        Returns:
            Formatted documentation string.
        """
        doc_parts: List[str] = []

        for idx, elem_id in enumerate(element_ids, start=1):
            doc = self.get_doc(elem_id)
            if doc is None:
                continue

            parts: List[str] = []
            parts.append(
                f"Documentation of UI element labeled with the numeric tag '{idx}':"
            )

            if doc.tap:
                parts.append(f"This UI element is clickable. {doc.tap}")
            if doc.text:
                parts.append(
                    f"This UI element can receive text input. "
                    f"The text input is used for: {doc.text}"
                )
            if doc.long_press:
                parts.append(f"This UI element is long clickable. {doc.long_press}")
            if doc.v_swipe:
                parts.append(
                    f"This element can be swiped vertically. {doc.v_swipe}"
                )
            if doc.h_swipe:
                parts.append(
                    f"This element can be swiped horizontally. {doc.h_swipe}"
                )

            if len(parts) > 1:  # More than just the header
                doc_parts.append("\n".join(parts))

        return "\n\n".join(doc_parts)

    def list_all(self) -> Dict[str, ElementDoc]:
        """
        List all documented elements.

        Returns:
            Dictionary mapping element_id to ElementDoc.
        """
        docs = {}
        for doc_file in self.docs_dir.glob("*.json"):
            element_id = doc_file.stem
            doc = self.get_doc(element_id)
            if doc:
                docs[element_id] = doc
        return docs

    @property
    def count(self) -> int:
        """Return the number of documented elements."""
        return len(list(self.docs_dir.glob("*.json")))
