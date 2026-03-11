"""
Documentation generator from human demonstrations.

Generates UI element documentation by analyzing before/after screenshots
from recorded human demonstrations. Refactored from the original
AppAgent's document_generation.py.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from agents.base_agent import BaseAgent, AgentError
from core.knowledge_base import KnowledgeBase
from core.logger import logger
from core.prompts import PromptBuilder


class DocGeneratorAgent(BaseAgent):
    """
    Documentation generator from human demos.

    Reads recorded demonstration steps and generates documentation
    for each UI element interaction using the LLM.
    """

    def __init__(
        self,
        app_name: str,
        demo_name: str,
        root_dir: str | Path = "./",
        voice_enabled: bool = False,
        config_path: str | Path = "config.yaml",
    ) -> None:
        super().__init__(app_name, root_dir, voice_enabled, config_path)
        self.demo_name = demo_name

    def _execute(self) -> None:
        """Generate documentation from recorded demo steps."""
        # Locate demo files
        app_dir = self.root_dir / "apps" / self.app_name
        demo_dir = app_dir / "demos" / self.demo_name
        labeled_ss_dir = demo_dir / "labeled_screenshots"
        record_path = demo_dir / "record.txt"
        task_desc_path = demo_dir / "task_desc.txt"
        log_path = demo_dir / f"log_{self.app_name}_{self.demo_name}.txt"

        # Validate paths
        for path in (demo_dir, labeled_ss_dir, record_path, task_desc_path):
            if not path.exists():
                raise AgentError(f"Required path not found: {path}")

        # Setup knowledge base
        docs_dir = app_dir / "demo_docs"
        knowledge_base = KnowledgeBase(docs_dir)

        # Read task description
        task_desc = task_desc_path.read_text().strip()
        logger.info(
            f"Generating docs for app '{self.app_name}' from demo '{self.demo_name}'"
        )

        # Read records
        records = record_path.read_text().strip().split("\n")
        total_steps = sum(1 for r in records if r.strip() != "stop")

        doc_count = 0
        for i, record in enumerate(records, start=1):
            record = record.strip()
            if record == "stop" or not record:
                continue

            # Parse the record
            action_str, resource_id = record.split(":::")
            action_type = action_str.split("(")[0]
            action_param = re.findall(r"\((.*?)\)", action_str)[0]

            # Map swipe directions to action types
            actual_action_type = action_type
            if action_type == "swipe":
                _, swipe_dir = action_param.split(":sep:")
                if swipe_dir in ("up", "down"):
                    actual_action_type = "v_swipe"
                elif swipe_dir in ("left", "right"):
                    actual_action_type = "h_swipe"

            # Check if documentation already exists
            if knowledge_base.has_doc(resource_id, actual_action_type):
                if not self.config.agent.doc_refine:
                    logger.info(
                        f"Doc already exists for {resource_id} ({actual_action_type}). "
                        f"Enable DOC_REFINE to update."
                    )
                    continue

            # Build prompt
            old_doc = ""
            if self.config.agent.doc_refine:
                existing = knowledge_base.get_doc(resource_id)
                if existing:
                    old_doc = getattr(existing, actual_action_type, "")

            # Get element number from action param
            elem_id = action_param.split(":sep:")[0] if ":sep:" in action_param else action_param

            prompt = PromptBuilder.build_doc_prompt(
                action_type, elem_id, task_desc, old_doc
            )

            # Images
            img_before = labeled_ss_dir / f"{self.demo_name}_{i}.png"
            img_after = labeled_ss_dir / f"{self.demo_name}_{i + 1}.png"

            if not img_before.exists() or not img_after.exists():
                logger.warning(f"Missing images for step {i}, skipping")
                continue

            # Get LLM response
            logger.info(
                f"Generating doc for {resource_id} ({actual_action_type}) "
                f"[{i}/{total_steps}]..."
            )
            try:
                response = self.model.get_response_with_retry(
                    prompt, [str(img_before), str(img_after)],
                    max_retries=self.config.agent.max_retries,
                )

                # Save documentation
                knowledge_base.save_doc(
                    resource_id, actual_action_type, response.content
                )

                # Log the interaction
                log_item = {
                    "step": i,
                    "prompt": prompt,
                    "image_before": img_before.name,
                    "image_after": img_after.name,
                    "response": response.content,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_item) + "\n")

                doc_count += 1

            except Exception as e:
                logger.error(f"Failed to generate doc for step {i}: {e}")

            time.sleep(self.config.agent.request_interval)

        logger.info(
            f"✓ Documentation generation complete. {doc_count} docs generated."
        )
