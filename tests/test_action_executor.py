"""Tests for the action executor — parsing LLM responses into actions."""

import pytest

from core.action_executor import (
    ActionParser,
    TapAction,
    TextAction,
    LongPressAction,
    SwipeAction,
    GridAction,
    FinishAction,
    ErrorAction,
    TapGridAction,
    SwipeGridAction,
)


class TestActionParser:
    """Tests for ActionParser.parse_explore_response."""

    def test_parse_tap_action(self):
        response = (
            "Observation: I see a home button\n"
            "Thought: I should tap the home button\n"
            "Action: tap(3)\n"
            "Summary: Tapped the home button"
        )
        action = ActionParser.parse_explore_response(response)
        assert isinstance(action, TapAction)
        assert action.element_index == 3
        assert action.summary == "Tapped the home button"

    def test_parse_text_action(self):
        response = (
            'Observation: I see a search field\n'
            'Thought: I should type a query\n'
            'Action: text("Hello world")\n'
            'Summary: Typed a search query'
        )
        action = ActionParser.parse_explore_response(response)
        assert isinstance(action, TextAction)
        assert action.text == "Hello world"

    def test_parse_swipe_action(self):
        response = (
            "Observation: I see a scroll view\n"
            "Thought: I should scroll down\n"
            'Action: swipe(5, "down", "medium")\n'
            "Summary: Scrolled down the list"
        )
        action = ActionParser.parse_explore_response(response)
        assert isinstance(action, SwipeAction)
        assert action.element_index == 5
        assert action.direction == "down"
        assert action.distance == "medium"

    def test_parse_long_press_action(self):
        response = (
            "Observation: I see a message\n"
            "Thought: I should long press it\n"
            "Action: long_press(7)\n"
            "Summary: Long pressed the message"
        )
        action = ActionParser.parse_explore_response(response)
        assert isinstance(action, LongPressAction)
        assert action.element_index == 7

    def test_parse_grid_action(self):
        response = (
            "Observation: No labeled elements match\n"
            "Thought: I need to use grid mode\n"
            "Action: grid()\n"
            "Summary: Activating grid overlay"
        )
        action = ActionParser.parse_explore_response(response)
        assert isinstance(action, GridAction)

    def test_parse_finish_action(self):
        response = (
            "Observation: Task appears done\n"
            "Thought: The task is complete\n"
            "Action: FINISH\n"
            "Summary: Completed the task"
        )
        action = ActionParser.parse_explore_response(response)
        assert isinstance(action, FinishAction)

    def test_parse_malformed_response(self):
        response = "This is not a valid response format"
        action = ActionParser.parse_explore_response(response)
        assert isinstance(action, ErrorAction)


class TestGridParser:
    """Tests for ActionParser.parse_grid_response."""

    def test_parse_grid_tap(self):
        response = (
            "Observation: I see the grid\n"
            "Thought: I should tap area 12\n"
            'Action: tap(12, "center")\n'
            "Summary: Tapped center of area 12"
        )
        action = ActionParser.parse_grid_response(response)
        assert isinstance(action, TapGridAction)
        assert action.area == 12
        assert action.subarea == "center"

    def test_parse_grid_swipe(self):
        response = (
            "Observation: I see the grid\n"
            "Thought: I should swipe\n"
            'Action: swipe(10, "center", 20, "center")\n'
            "Summary: Swiped from area 10 to 20"
        )
        action = ActionParser.parse_grid_response(response)
        assert isinstance(action, SwipeGridAction)
        assert action.start_area == 10
        assert action.end_area == 20


class TestReflectionParser:
    """Tests for ActionParser.parse_reflection."""

    def test_parse_success(self):
        response = (
            "Decision: SUCCESS\n"
            "Thought: The action moved the task forward\n"
            "Documentation: This button opens the settings menu"
        )
        result = ActionParser.parse_reflection(response)
        assert result.decision == "SUCCESS"
        assert "settings menu" in result.documentation

    def test_parse_ineffective(self):
        response = (
            "Decision: INEFFECTIVE\n"
            "Thought: Nothing changed on screen"
        )
        result = ActionParser.parse_reflection(response)
        assert result.decision == "INEFFECTIVE"
        assert result.documentation == ""

    def test_parse_back(self):
        response = (
            "Decision: BACK\n"
            "Thought: This navigated to the wrong page\n"
            "Documentation: This element navigates to account settings"
        )
        result = ActionParser.parse_reflection(response)
        assert result.decision == "BACK"
        assert "account settings" in result.documentation
