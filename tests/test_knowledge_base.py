"""Tests for the knowledge base."""

import json
from pathlib import Path

import pytest

from core.knowledge_base import KnowledgeBase, ElementDoc


@pytest.fixture
def kb(tmp_path):
    """Create a KnowledgeBase with a temporary directory."""
    return KnowledgeBase(tmp_path / "docs")


class TestKnowledgeBase:
    def test_save_and_get_doc(self, kb):
        kb.save_doc("btn_home", "tap", "Opens the home screen")
        doc = kb.get_doc("btn_home")
        assert doc is not None
        assert doc.tap == "Opens the home screen"

    def test_get_nonexistent_doc(self, kb):
        assert kb.get_doc("nonexistent") is None

    def test_update_existing_doc(self, kb):
        kb.save_doc("btn_home", "tap", "Opens home")
        kb.save_doc("btn_home", "long_press", "Shows context menu")

        doc = kb.get_doc("btn_home")
        assert doc.tap == "Opens home"
        assert doc.long_press == "Shows context menu"

    def test_has_doc(self, kb):
        kb.save_doc("btn_search", "tap", "Opens search")

        assert kb.has_doc("btn_search") is True
        assert kb.has_doc("btn_search", "tap") is True
        assert kb.has_doc("btn_search", "long_press") is False
        assert kb.has_doc("nonexistent") is False

    def test_get_docs_for_elements(self, kb):
        kb.save_doc("btn_home", "tap", "Opens the home screen")
        kb.save_doc("btn_search", "tap", "Opens search bar")

        doc_str = kb.get_docs_for_elements(["btn_home", "btn_search", "unknown"])
        assert "home screen" in doc_str
        assert "search bar" in doc_str

    def test_list_all(self, kb):
        kb.save_doc("btn_a", "tap", "Doc A")
        kb.save_doc("btn_b", "tap", "Doc B")

        all_docs = kb.list_all()
        assert len(all_docs) == 2
        assert "btn_a" in all_docs
        assert "btn_b" in all_docs

    def test_count(self, kb):
        assert kb.count == 0
        kb.save_doc("btn_a", "tap", "Doc A")
        assert kb.count == 1
        kb.save_doc("btn_b", "tap", "Doc B")
        assert kb.count == 2

    def test_stores_as_json(self, kb):
        kb.save_doc("btn_test", "tap", "Test doc")
        doc_file = kb.docs_dir / "btn_test.json"
        assert doc_file.exists()

        data = json.loads(doc_file.read_text())
        assert data["tap"] == "Test doc"
        assert data["text"] == ""
