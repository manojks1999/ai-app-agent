"""
Prompt templates for the agent.

Organized as a PromptBuilder class with clean string formatting.
Replaces the original's raw string templates with a structured approach.
"""

from __future__ import annotations


class PromptBuilder:
    """Builds prompts for the LLM agent across all phases."""

    # ─── Task Execution Prompts ──────────────────────────────────────────

    @staticmethod
    def build_task_prompt(
        task_desc: str,
        last_action: str,
        ui_document: str = "",
    ) -> str:
        """Build the main task execution prompt (element mode)."""
        doc_section = ""
        if ui_document:
            doc_section = f"""
You also have access to the following documentations that describe the functionalities of UI
elements you can interact with on the screen. These docs are crucial for you to determine the
target of your next action. You should always prioritize these documented elements for interaction:
{ui_document}"""

        return f"""You are an agent that is trained to perform some basic tasks on a smartphone. You will be given a \
smartphone screenshot. The interactive UI elements on the screenshot are labeled with numeric tags starting from 1. \
The numeric tag of each interactive element is located in the center of the element.

You can call the following functions to control the smartphone:

1. tap(element: int)
This function is used to tap a UI element shown on the smartphone screen.
"element" is a numeric tag assigned to a UI element shown on the smartphone screen.
A simple use case can be tap(5), which taps the UI element labeled with the number 5.

2. text(text_input: str)
This function is used to insert text input in an input field/box. text_input is the string you want to insert and \
must be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string \
"Hello, world!" into the input area on the smartphone screen. This function is usually callable when you see a \
keyboard showing in the lower half of the screen.

3. long_press(element: int)
This function is used to long press a UI element shown on the smartphone screen.
"element" is a numeric tag assigned to a UI element shown on the smartphone screen.
A simple use case can be long_press(5), which long presses the UI element labeled with the number 5.

4. swipe(element: int, direction: str, dist: str)
This function is used to swipe a UI element shown on the smartphone screen, usually a scroll view or a slide bar.
"element" is a numeric tag assigned to a UI element shown on the smartphone screen. "direction" is a string that \
represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation \
marks. "dist" determines the distance of the swipe and can be one of the three options: short, medium, long.
A simple use case can be swipe(21, "up", "medium"), which swipes up the UI element labeled with the number 21 for \
a medium distance.

5. grid()
You should call this function when you find the element you want to interact with is not labeled with a numeric tag \
and other elements with numeric tags cannot help with the task. The function will bring up a grid overlay to divide \
the smartphone screen into small areas and this will give you more freedom to choose any part of the screen to tap, \
long press, or swipe.
{doc_section}
The task you need to complete is to {task_desc}. Your past actions to proceed with this task are summarized as \
follows: {last_action}
Now, given the documentation and the following labeled screenshot, you need to think and call the function needed to \
proceed with the task. Your output should include four parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task. If you believe the task is completed \
or there is nothing to be done, you should output FINISH. You cannot output anything else except a function call or \
FINISH in this field.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the \
numeric tag in your summary>
You can only take one action at a time, so please directly call the function."""

    @staticmethod
    def build_task_prompt_grid(task_desc: str, last_action: str) -> str:
        """Build the task execution prompt for grid mode."""
        return f"""You are an agent that is trained to perform some basic tasks on a smartphone. You will be given a \
smartphone screenshot overlaid by a grid. The grid divides the screenshot into small square areas. Each area is \
labeled with an integer in the top-left corner.

You can call the following functions to control the smartphone:

1. tap(area: int, subarea: str)
This function is used to tap a grid area shown on the smartphone screen. "area" is the integer label assigned to a \
grid area. "subarea" is a string representing the exact location to tap within the grid area. It can take one of the \
nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be tap(5, "center"), which taps the exact center of the grid area labeled with the number 5.

2. long_press(area: int, subarea: str)
This function is used to long press a grid area shown on the smartphone screen. "area" is the integer label assigned \
to a grid area. "subarea" is a string representing the exact location to long press within the grid area.
A simple use case can be long_press(7, "top-left"), which long presses the top left part of the grid area 7.

3. swipe(start_area: int, start_subarea: str, end_area: int, end_subarea: str)
This function is used to perform a swipe action on the smartphone screen. "start_area" and "start_subarea" mark the \
starting location; "end_area" and "end_subarea" mark the ending location. The two subarea parameters can take one of \
the nine values mentioned above.
A simple use case can be swipe(21, "center", 25, "right"), which performs a swipe from the center of area 21 to the \
right part of area 25.

The task you need to complete is to {task_desc}. Your past actions to proceed with this task are summarized as \
follows: {last_action}
Now, given the following labeled screenshot, you need to think and call the function needed to proceed with the task. \
Your output should include four parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task. If you believe the task is completed \
or there is nothing to be done, you should output FINISH. You cannot output anything else except a function call or \
FINISH in this field.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the grid \
area number in your summary>
You can only take one action at a time, so please directly call the function."""

    # ─── Exploration Prompts ─────────────────────────────────────────────

    @staticmethod
    def build_explore_prompt(task_desc: str, last_action: str) -> str:
        """Build the autonomous exploration prompt."""
        return f"""You are an agent that is trained to complete certain tasks on a smartphone. You will be given a \
screenshot of a smartphone app. The interactive UI elements on the screenshot are labeled with numeric tags starting \
from 1.

You can call the following functions to interact with those labeled elements to control the smartphone:

1. tap(element: int)
This function is used to tap a UI element shown on the smartphone screen.
"element" is a numeric tag assigned to a UI element shown on the smartphone screen.
A simple use case can be tap(5), which taps the UI element labeled with the number 5.

2. text(text_input: str)
This function is used to insert text input in an input field/box. text_input is the string you want to insert and \
must be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string \
"Hello, world!" into the input area on the smartphone screen. This function is only callable when you see a keyboard \
showing in the lower half of the screen.

3. long_press(element: int)
This function is used to long press a UI element shown on the smartphone screen.
"element" is a numeric tag assigned to a UI element shown on the smartphone screen.
A simple use case can be long_press(5), which long presses the UI element labeled with the number 5.

4. swipe(element: int, direction: str, dist: str)
This function is used to swipe a UI element shown on the smartphone screen, usually a scroll view or a slide bar.
"element" is a numeric tag assigned to a UI element shown on the smartphone screen. "direction" is a string that \
represents one of the four directions: up, down, left, right. "dist" determines the distance of the swipe and can be \
one of the three options: short, medium, long.
A simple use case can be swipe(21, "up", "medium"), which swipes up the UI element labeled with the number 21 for a \
medium distance.

The task you need to complete is to {task_desc}. Your past actions to proceed with this task are summarized as \
follows: {last_action}
Now, given the following labeled screenshot, you need to think and call the function needed to proceed with the task. \
Your output should include four parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task. If you believe the task is completed \
or there is nothing to be done, you should output FINISH. You cannot output anything else except a function call or \
FINISH in this field.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the \
numeric tag in your summary>
You can only take one action at a time, so please directly call the function."""

    # ─── Reflection Prompts ──────────────────────────────────────────────

    @staticmethod
    def build_reflect_prompt(
        action_type: str,
        element_id: str,
        task_desc: str,
        last_action: str,
    ) -> str:
        """Build the self-reflection prompt after an action."""
        action_verb_map = {
            "tap": "tapping",
            "long_press": "long pressing",
            "swipe": "swiping",
        }
        action_verb = action_verb_map.get(action_type, action_type)

        return f"""I will give you screenshots of a mobile app before and after {action_verb} the UI element labeled \
with the number '{element_id}' on the first screenshot. The numeric tag of each element is located at the center of \
the element. The action of {action_verb} this UI element was described as follows:
{last_action}
The action was also an attempt to proceed with a larger task, which is to {task_desc}. Your job is to carefully \
analyze the difference between the two screenshots to determine if the action is in accord with the description above \
and at the same time effectively moved the task forward. Your output should be determined based on the following \
situations:
1. BACK
If you think the action navigated you to a page where you cannot proceed with the given task, you should go back to \
the previous interface. At the same time, describe the functionality of the UI element concisely in one or two \
sentences. Never include the numeric tag in your description.
Decision: BACK
Thought: <explain why you think the last action is wrong>
Documentation: <describe the function of the UI element>
2. INEFFECTIVE
If you find the action changed nothing on the screen (screenshots before and after are identical), you should \
continue to interact with other elements.
Decision: INEFFECTIVE
Thought: <explain why you made this decision>
3. CONTINUE
If you find the action changed something but did not move the task forward, describe the element and continue.
Decision: CONTINUE
Thought: <explain why the action did not move the task forward>
Documentation: <describe the function of the UI element>
4. SUCCESS
If you think the action successfully moved the task forward, describe the element's functionality.
Decision: SUCCESS
Thought: <explain why the action was successful>
Documentation: <describe the function of the UI element>"""

    # ─── Documentation Prompts ───────────────────────────────────────────

    @staticmethod
    def build_doc_prompt(
        action_type: str,
        element_id: str,
        task_desc: str,
        old_doc: str = "",
    ) -> str:
        """Build a documentation generation prompt for a UI element."""
        action_verb_map = {
            "tap": ("tapping", "Tapping"),
            "text": ("typing in", "This input area"),
            "long_press": ("long pressing", "Long pressing"),
            "swipe": ("swiping", "Swiping"),
        }
        verb, prefix = action_verb_map.get(action_type, (action_type, action_type))

        prompt = f"""I will give you the screenshot of a mobile app before and after {verb} the UI element labeled \
with the number {element_id} on the screen. The numeric tag of each element is located at the center of the element. \
{prefix} this UI element is a necessary part of proceeding with a larger task, which is to {task_desc}. Your task is \
to describe the functionality of the UI element concisely in one or two sentences. Notice that your description of the \
UI element should focus on the general function. Never include the numeric tag of the UI element in your description. \
You can use pronouns such as "the UI element" to refer to the element."""

        if old_doc:
            prompt += f"""
A documentation of this UI element generated from previous demos is shown below. Your generated description should be \
based on this previous doc and optimize it. Notice that it is possible that your understanding of the function of the \
UI element derived from the given screenshots conflicts with the previous doc, because the function of a UI element \
can be flexible. In this case, your generated description should combine both.
Old documentation of this UI element: {old_doc}"""

        return prompt
