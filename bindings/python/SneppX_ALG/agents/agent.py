"""A minimal ReAct-style agent loop driven by a user-supplied policy."""

from dataclasses import dataclass
from typing import Callable, List, Optional

from .memory import ConversationMemory
from .messages import Conversation
from .parsing import parse
from .tools import ToolRegistry


@dataclass
class Step:
    """One reasoning step of an agent run."""

    index: int
    thought: Optional[str]
    action: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]
    final: Optional[str]


class Agent:
    """Runs a policy (any ``callable(Conversation) -> str``) in a ReAct loop.

    The policy emits ``Thought``/``Action``/``Action Input`` blocks to call a
    registered tool, or ``Final Answer: ...`` to finish. This keeps the loop
    fully testable without an LLM.
    """

    def __init__(
        self,
        policy: Callable[[Conversation], str],
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        max_steps: int = 6,
        name: str = "agent",
    ):
        self.policy = policy
        self.tools = tools if tools is not None else ToolRegistry()
        self.memory = memory if memory is not None else ConversationMemory()
        self.max_steps = int(max_steps)
        self.name = name
        self.steps: List[Step] = []

    def run(self, task: str) -> str:
        """Execute the ReAct loop for ``task`` and return the final answer."""
        self.steps = []
        self.memory.add("user", task)
        for i in range(self.max_steps):
            output = self.policy(self.memory.conversation)
            self.memory.add("assistant", output)
            parsed = parse(output)
            if parsed["final"] is not None:
                self.steps.append(
                    Step(i, parsed["thought"], None, None, None, parsed["final"])
                )
                return parsed["final"]
            action = parsed["action"]
            if not action:
                self.steps.append(Step(i, parsed["thought"], None, None, None, output))
                return output
            tool_input = parsed["action_input"] or ""
            try:
                observation = str(self.tools.call(action, tool_input))
            except KeyError:
                observation = f"Error: unknown tool {action!r}"
            self.memory.add("observation", observation)
            self.steps.append(
                Step(i, parsed["thought"], action, tool_input, observation, None)
            )
        return "max_steps exceeded"

    def last_step(self) -> Optional[Step]:
        return self.steps[-1] if self.steps else None
