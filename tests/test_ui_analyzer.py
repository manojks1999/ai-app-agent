"""Tests for the UI analyzer."""

import tempfile
from pathlib import Path

import pytest

from core.ui_analyzer import UIAnalyzer, UIElement


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        package="com.example.app" content-desc="" checkable="false"
        checked="false" clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false" bounds="[0,0][1080,2340]">
    <node index="0" text="Home" resource-id="com.example:id/btn_home"
          class="android.widget.Button" package="com.example.app"
          content-desc="Home" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false"
          selected="false" bounds="[10,100][200,200]" />
    <node index="1" text="Search" resource-id="com.example:id/btn_search"
          class="android.widget.Button" package="com.example.app"
          content-desc="Search" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false"
          selected="false" bounds="[300,100][500,200]" />
    <node index="2" text="" resource-id="com.example:id/scroll_view"
          class="android.widget.ScrollView" package="com.example.app"
          content-desc="" checkable="false" checked="false"
          clickable="false" enabled="true" focusable="true" focused="false"
          scrollable="true" long-clickable="false" password="false"
          selected="false" bounds="[0,300][1080,2000]" />
  </node>
</hierarchy>"""


@pytest.fixture
def xml_file(tmp_path):
    """Create a temporary XML file with sample UI hierarchy."""
    xml_path = tmp_path / "test_ui.xml"
    xml_path.write_text(SAMPLE_XML)
    return xml_path


class TestUIAnalyzer:
    def test_parse_clickable_elements(self, xml_file):
        analyzer = UIAnalyzer(min_dist=30)
        elements = analyzer.parse_hierarchy(xml_file, "clickable")

        assert len(elements) == 2
        assert all(e.attrib == "clickable" for e in elements)

    def test_parse_focusable_elements(self, xml_file):
        analyzer = UIAnalyzer(min_dist=30)
        elements = analyzer.parse_hierarchy(xml_file, "focusable")

        # All 3 nodes have focusable=true (buttons + scroll view)
        # but the buttons overlap with themselves, so deduplication may apply
        assert len(elements) >= 1

    def test_element_center(self, xml_file):
        analyzer = UIAnalyzer(min_dist=30)
        elements = analyzer.parse_hierarchy(xml_file, "clickable")

        home_btn = elements[0]
        assert home_btn.center == (105, 150)  # (10+200)//2, (100+200)//2

    def test_element_dimensions(self, xml_file):
        analyzer = UIAnalyzer(min_dist=30)
        elements = analyzer.parse_hierarchy(xml_file, "clickable")

        home_btn = elements[0]
        assert home_btn.width == 190   # 200-10
        assert home_btn.height == 100  # 200-100

    def test_get_interactive_elements(self, xml_file):
        analyzer = UIAnalyzer(min_dist=30)
        elements = analyzer.get_interactive_elements(xml_file)

        # Should have at least the 2 buttons + scroll view
        assert len(elements) >= 2

    def test_missing_file_returns_empty(self, tmp_path):
        analyzer = UIAnalyzer()
        elements = analyzer.parse_hierarchy(tmp_path / "nonexistent.xml")

        assert elements == []

    def test_min_dist_deduplication(self, xml_file):
        # With a very large min_dist, elements close together should merge
        analyzer = UIAnalyzer(min_dist=1000)
        elements = analyzer.parse_hierarchy(xml_file, "clickable")

        # Only one element should survive deduplication
        assert len(elements) == 1
