"""
Task execution agent for the deployment phase.

Executes tasks on an app using previously generated documentation.
Supports both element-mode and grid-mode interactions. Refactored
from the original AppAgent's task_executor.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agents.base_agent import BaseAgent, AgentError
from core.action_executor import (
    ActionExecutor,
    ActionParser,
    FinishAction,
    GridAction,
    ErrorAction,
    TapGridAction,
    LongPressGridAction,
    SwipeGridAction,
)
from core.image_processor import ImageProcessor
from core.knowledge_base import KnowledgeBase
from core.logger import logger, print_colored
from core.prompts import PromptBuilder
from core.ui_analyzer import UIAnalyzer


class TaskRunnerAgent(BaseAgent):
    """
    Task execution agent (deployment phase).

    Uses previously generated documentation to execute tasks on an app.
    Supports both labeled-element mode and grid overlay mode.
    """

    def _execute(self) -> None:
        """Run the task execution loop."""
        # Setup directories
        app_dir = self.root_dir / "apps" / self.app_name
        tasks_dir = self.root_dir / "tasks"
        task_name = self._generate_timestamp_name("task")
        task_dir = tasks_dir / task_name
        self._create_work_dirs(task_dir)

        # Locate documentation
        auto_docs_dir = app_dir / "auto_docs"
        demo_docs_dir = app_dir / "demo_docs"
        knowledge_base, no_doc = self._select_docs(auto_docs_dir, demo_docs_dir)

        # Initialize components
        ui_analyzer = UIAnalyzer(min_dist=self.config.agent.min_dist)
        image_processor = ImageProcessor()
        action_executor = ActionExecutor(
            self.controller,
            self.controller.width,
            self.controller.height,
        )

        # Log file
        log_path = task_dir / f"log_{task_name}.txt"

        # Get task description
        task_desc = self._get_task_description()
        logger.info(f"Task: {task_desc}")

        # Task execution loop
        round_count = 0
        last_action_summary = "None"
        task_complete = False
        grid_on = False
        grid_rows, grid_cols = 0, 0

        while round_count < self.config.agent.max_rounds:
            round_count += 1
            logger.info(f"═══ Round {round_count} ═══")

            try:
                # Capture current state
                ss_prefix = f"{task_name}_{round_count}"
                screenshot = self.controller.get_screenshot(ss_prefix, task_dir)
                xml_path = self.controller.get_xml(ss_prefix, task_dir)

                if grid_on:
                    # Grid mode
                    grid_path = task_dir / f"{ss_prefix}_grid.png"
                    grid_rows, grid_cols = image_processor.draw_grid(
                        screenshot, grid_path
                    )
                    image = grid_path
                    prompt = PromptBuilder.build_task_prompt_grid(
                        task_desc, last_action_summary
                    )
                else:
                    # Element mode
                    elements = ui_analyzer.get_interactive_elements(xml_path)
                    labeled_path = task_dir / f"{ss_prefix}_labeled.png"
                    image_processor.label_elements(
                        screenshot, labeled_path, elements,
                        dark_mode=self.config.agent.dark_mode,
                    )
                    image = labeled_path

                    # Build prompt with documentation
                    ui_doc = ""
                    if knowledge_base and not no_doc:
                        elem_ids = [e.uid for e in elements]
                        ui_doc = knowledge_base.get_docs_for_elements(elem_ids)
                        if ui_doc:
                            logger.info(f"Documentation retrieved for current UI")

                    prompt = PromptBuilder.build_task_prompt(
                        task_desc, last_action_summary, ui_doc
                    )

                # Get LLM response
                logger.info("Thinking about next action...")
                response = self.model.get_response_with_retry(
                    prompt, [str(image)],
                    max_retries=self.config.agent.max_retries,
                    backoff_factor=self.config.agent.retry_backoff_factor,
                )

                # Log the interaction
                log_item = {
                    "step": round_count,
                    "prompt": prompt,
                    "image": str(image.name),
                    "response": response.content,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_item) + "\n")

                # Parse response
                if grid_on:
                    action = ActionParser.parse_grid_response(response.content)
                else:
                    action = ActionParser.parse_explore_response(response.content)

                if isinstance(action, FinishAction):
                    task_complete = True
                    break

                if isinstance(action, ErrorAction):
                    logger.error(f"Parse error: {action.error_message}")
                    break

                last_action_summary = action.summary

                # Handle grid toggle
                if isinstance(action, GridAction):
                    grid_on = True
                    continue

                # Execute the action
                if grid_on:
                    action_executor.execute(
                        action, grid_rows=grid_rows, grid_cols=grid_cols
                    )
                else:
                    action_executor.execute(action, elements=elements)

                # Exit grid mode after non-grid action
                if not isinstance(action, GridAction):
                    grid_on = False

            except Exception as e:
                logger.error(f"Error in round {round_count}: {e}")
                break

            time.sleep(self.config.agent.request_interval)

        # Summary
        if task_complete:
            logger.info("✓ Task completed successfully")
        elif round_count >= self.config.agent.max_rounds:
            logger.warning(
                f"Task reached max rounds ({self.config.agent.max_rounds})"
            )
        else:
            logger.error("Task ended unexpectedly")

    def _select_docs(
        self,
        auto_docs_dir: Path,
        demo_docs_dir: Path,
    ) -> tuple[KnowledgeBase | None, bool]:
        """
        Select which documentation base to use.

        Returns:
            Tuple of (KnowledgeBase or None, no_doc flag).
        """
        has_auto = auto_docs_dir.exists() and any(auto_docs_dir.glob("*.json"))
        has_demo = demo_docs_dir.exists() and any(demo_docs_dir.glob("*.json"))

        if not has_auto and not has_demo:
            choice = self.voice.get_choice(
                f"No documentation found for '{self.app_name}'. "
                "Proceed without docs? (y/n)",
                ["y", "n"],
            )
            if choice == "y":
                return None, True
            raise AgentError("No documentation available and user declined to proceed")

        if has_auto and has_demo:
            choice = self.voice.get_choice(
                f"Documentation available from both sources:\n"
                "  1. Autonomous exploration\n"
                "  2. Human demonstration\n"
                "Choose 1 or 2:",
                ["1", "2"],
            )
            docs_dir = auto_docs_dir if choice == "1" else demo_docs_dir
        elif has_auto:
            logger.info("Using auto-exploration documentation")
            docs_dir = auto_docs_dir
        else:
            logger.info("Using human demonstration documentation")
            docs_dir = demo_docs_dir

        return KnowledgeBase(docs_dir), False
