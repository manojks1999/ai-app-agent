"""
Human demonstration recorder.

Records user demonstrations on the device by showing labeled screenshots
and accepting voice/keyboard commands for actions. Refactored from the
original AppAgent's step_recorder.py.
"""

from __future__ import annotations

import time
from pathlib import Path

from agents.base_agent import BaseAgent, AgentError
from core.image_processor import ImageProcessor
from core.logger import logger, print_colored
from core.ui_analyzer import UIAnalyzer


class DemoRecorderAgent(BaseAgent):
    """
    Human demonstration recorder.

    Shows labeled screenshots and accepts voice/keyboard commands
    for the user to demonstrate how to complete a task. Records all
    actions for later documentation generation.
    """

    def _execute(self) -> None:
        """Run the demonstration recording loop."""
        # Setup directories
        app_dir = self.root_dir / "apps" / self.app_name
        demo_dir = app_dir / "demos"
        demo_name = self._generate_timestamp_name("demo")
        task_dir = demo_dir / demo_name
        raw_ss_dir = task_dir / "raw_screenshots"
        xml_dir = task_dir / "xml"
        labeled_ss_dir = task_dir / "labeled_screenshots"
        self._create_work_dirs(task_dir, raw_ss_dir, xml_dir, labeled_ss_dir)

        # Record file
        record_path = task_dir / "record.txt"
        task_desc_path = task_dir / "task_desc.txt"

        # Initialize components
        ui_analyzer = UIAnalyzer(min_dist=self.config.agent.min_dist)
        image_processor = ImageProcessor()

        # Get task description
        task_desc = self._get_task_description()
        task_desc_path.write_text(task_desc)
        logger.info(f"Demo task: {task_desc}")

        print_colored(
            "All interactive elements are labeled with numbered tags.\n"
            "Red tags = clickable, Blue tags = scrollable.\n"
            "Use voice or type commands: tap, text, long press, swipe, stop",
            "blue",
        )

        step = 0
        records: list[str] = []

        while True:
            step += 1
            try:
                # Capture current state
                screenshot = self.controller.get_screenshot(
                    f"{demo_name}_{step}", raw_ss_dir
                )
                xml_path = self.controller.get_xml(f"{demo_name}_{step}", xml_dir)

                # Analyze and label
                elements = ui_analyzer.get_interactive_elements(xml_path)
                labeled_path = labeled_ss_dir / f"{demo_name}_{step}.png"
                labeled_img = image_processor.label_elements(
                    screenshot, labeled_path, elements, record_mode=True
                )

                # Show the labeled screenshot
                try:
                    import cv2
                    cv2.imshow("VoiceAppAgent Demo", labeled_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except Exception:
                    logger.info(f"Screenshot saved: {labeled_path}")
                    print_colored(
                        f"View screenshot at: {labeled_path}", "yellow"
                    )

                # Get action from user
                action_input = self.voice.get_choice(
                    f"Choose action (step {step}):",
                    ["tap", "text", "long press", "swipe", "stop"],
                )

                if action_input == "stop":
                    records.append("stop")
                    break

                elif action_input == "tap":
                    elem_num = self._get_element_number(len(elements))
                    elem = elements[elem_num - 1]
                    x, y = elem.center
                    self.controller.tap(x, y)
                    records.append(f"tap({elem_num}):::{elem.uid}")

                elif action_input == "text":
                    elem_num = self._get_element_number(len(elements))
                    text = self.voice.get_input("Enter the text to type:")
                    self.controller.text(text)
                    elem = elements[elem_num - 1]
                    records.append(
                        f'text({elem_num}:sep:"{text}"):::{elem.uid}'
                    )

                elif action_input == "long press":
                    elem_num = self._get_element_number(len(elements))
                    elem = elements[elem_num - 1]
                    x, y = elem.center
                    self.controller.long_press(x, y)
                    records.append(f"long_press({elem_num}):::{elem.uid}")

                elif action_input == "swipe":
                    direction = self.voice.get_choice(
                        "Swipe direction:", ["up", "down", "left", "right"]
                    )
                    elem_num = self._get_element_number(len(elements))
                    elem = elements[elem_num - 1]
                    x, y = elem.center
                    self.controller.swipe(x, y, direction)
                    records.append(
                        f"swipe({elem_num}:sep:{direction}):::{elem.uid}"
                    )

            except Exception as e:
                logger.error(f"Error in demo step {step}: {e}")
                break

            time.sleep(3)

        # Save records
        record_path.write_text("\n".join(records) + "\n")
        logger.info(f"✓ Demo recording complete. {step} steps recorded to {record_path}")

        # Store demo_name for doc generation
        self._last_demo_name = demo_name

    def _get_element_number(self, max_elements: int) -> int:
        """Get a valid element number from the user."""
        while True:
            response = self.voice.get_input(
                f"Which element? (1 to {max_elements}):"
            )
            try:
                num = int(response)
                if 1 <= num <= max_elements:
                    return num
            except ValueError:
                pass
            print_colored(f"Invalid. Enter a number from 1 to {max_elements}.", "red")
