"""
Autonomous exploration agent.

Explores an app to generate UI element documentation. Refactored from
the original AppAgent's self_explorer.py with clean separation of
concerns and voice input support.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Set

from agents.base_agent import BaseAgent, AgentError
from core.action_executor import (
    ActionExecutor,
    ActionParser,
    FinishAction,
    TapAction,
    LongPressAction,
    SwipeAction,
    TextAction,
    ErrorAction,
    ParsedAction,
)
from core.image_processor import ImageProcessor
from core.knowledge_base import KnowledgeBase
from core.logger import logger
from core.prompts import PromptBuilder
from core.ui_analyzer import UIAnalyzer


class ExplorerAgent(BaseAgent):
    """
    Autonomous exploration agent.

    Explores an app by interacting with UI elements and generating
    documentation for each. Uses reflection to evaluate whether
    actions were effective.
    """

    def _execute(self) -> None:
        """Run the autonomous exploration loop."""
        # Setup directories
        app_dir = self.root_dir / "apps" / self.app_name
        demo_dir = app_dir / "demos"
        task_name = self._generate_timestamp_name("explore")
        task_dir = demo_dir / task_name
        docs_dir = app_dir / "auto_docs"
        self._create_work_dirs(task_dir, docs_dir)

        # Initialize components
        ui_analyzer = UIAnalyzer(min_dist=self.config.agent.min_dist)
        image_processor = ImageProcessor()
        knowledge_base = KnowledgeBase(docs_dir)
        action_executor = ActionExecutor(
            self.controller,
            self.controller.width,
            self.controller.height,
        )

        # Log files
        explore_log = task_dir / f"log_explore_{task_name}.txt"
        reflect_log = task_dir / f"log_reflect_{task_name}.txt"

        # Get task description
        task_desc = self._get_task_description()
        logger.info(f"Exploration task: {task_desc}")

        # Exploration loop
        round_count = 0
        doc_count = 0
        useless_set: Set[str] = set()
        last_action_summary = "None"
        task_complete = False

        while round_count < self.config.agent.max_rounds:
            round_count += 1
            logger.info(f"═══ Round {round_count} ═══")

            try:
                # Capture state
                screenshot = self.controller.get_screenshot(
                    f"{round_count}_before", task_dir
                )
                xml_path = self.controller.get_xml(f"{round_count}", task_dir)

                # Analyze UI
                elements = ui_analyzer.get_interactive_elements(xml_path)
                # Filter out known useless elements
                elements = [e for e in elements if e.uid not in useless_set]

                # Label screenshot
                labeled_path = task_dir / f"{round_count}_before_labeled.png"
                image_processor.label_elements(
                    screenshot, labeled_path, elements,
                    dark_mode=self.config.agent.dark_mode,
                )

                # Build prompt and get LLM response
                prompt = PromptBuilder.build_explore_prompt(
                    task_desc, last_action_summary
                )
                logger.info("Thinking about next action...")
                response = self.model.get_response_with_retry(
                    prompt, [str(labeled_path)],
                    max_retries=self.config.agent.max_retries,
                    backoff_factor=self.config.agent.retry_backoff_factor,
                )

                # Log the interaction
                self._log_step(explore_log, round_count, prompt, labeled_path, response.content)

                # Parse the response
                action = ActionParser.parse_explore_response(response.content)
                last_action_summary = action.summary

                if isinstance(action, FinishAction):
                    task_complete = True
                    break

                if isinstance(action, ErrorAction):
                    logger.error(f"Parse error: {action.error_message}")
                    break

                # Execute the action
                action_executor.execute(action, elements)

                # Skip reflection for text actions
                if isinstance(action, TextAction):
                    time.sleep(self.config.agent.request_interval)
                    continue

                # Capture after-state
                screenshot_after = self.controller.get_screenshot(
                    f"{round_count}_after", task_dir
                )
                labeled_after = task_dir / f"{round_count}_after_labeled.png"
                image_processor.label_elements(
                    screenshot_after, labeled_after, elements,
                    dark_mode=self.config.agent.dark_mode,
                )

                # Determine action type for reflection
                action_type = action.name
                element_idx = 0
                if isinstance(action, TapAction):
                    element_idx = action.element_index
                elif isinstance(action, LongPressAction):
                    element_idx = action.element_index
                elif isinstance(action, SwipeAction):
                    element_idx = action.element_index
                    direction = action.direction
                    if direction in ("up", "down"):
                        action_type = "v_swipe"
                    elif direction in ("left", "right"):
                        action_type = "h_swipe"

                # Self-reflection
                reflect_prompt = PromptBuilder.build_reflect_prompt(
                    action.name, str(element_idx), task_desc, last_action_summary
                )
                logger.info("Reflecting on action...")
                reflect_response = self.model.get_response_with_retry(
                    reflect_prompt, [str(labeled_path), str(labeled_after)],
                    max_retries=self.config.agent.max_retries,
                )

                # Log reflection
                self._log_step(
                    reflect_log, round_count, reflect_prompt,
                    labeled_path, reflect_response.content
                )

                # Parse reflection
                reflection = ActionParser.parse_reflection(reflect_response.content)
                resource_id = elements[element_idx - 1].uid if element_idx > 0 else ""

                if reflection.decision == "ERROR":
                    logger.error(f"Reflection error: {reflection.thought}")
                    break
                elif reflection.decision == "INEFFECTIVE":
                    useless_set.add(resource_id)
                    last_action_summary = "None"
                elif reflection.decision in ("BACK", "CONTINUE"):
                    useless_set.add(resource_id)
                    last_action_summary = "None"
                    if reflection.decision == "BACK":
                        self.controller.back()
                    # Save documentation
                    if reflection.documentation and resource_id:
                        knowledge_base.save_doc(
                            resource_id, action_type, reflection.documentation
                        )
                        doc_count += 1
                elif reflection.decision == "SUCCESS":
                    if reflection.documentation and resource_id:
                        knowledge_base.save_doc(
                            resource_id, action_type, reflection.documentation
                        )
                        doc_count += 1

            except Exception as e:
                logger.error(f"Error in exploration round {round_count}: {e}")
                break

            time.sleep(self.config.agent.request_interval)

        # Summary
        if task_complete:
            logger.info(
                f"✓ Exploration completed successfully. {doc_count} docs generated."
            )
        elif round_count >= self.config.agent.max_rounds:
            logger.warning(
                f"Exploration reached max rounds ({self.config.agent.max_rounds}). "
                f"{doc_count} docs generated."
            )
        else:
            logger.error(f"Exploration ended unexpectedly. {doc_count} docs generated.")

    @staticmethod
    def _log_step(
        log_path: Path,
        step: int,
        prompt: str,
        image: Path,
        response: str,
    ) -> None:
        """Append a log entry for a step."""
        log_item = {
            "step": step,
            "prompt": prompt,
            "image": str(image.name),
            "response": response,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_item) + "\n")
