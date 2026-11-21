"""A lightweight, dependency-free agent framework.

Components: messages, a tool registry, ReAct-style output parsing, conversation
memory, and a policy-driven :class:`Agent` loop.
"""

from . import agent, memory, messages, parsing, tools
from .agent import Agent, Step
from .memory import ConversationMemory
from .messages import ROLES, Conversation, Message
from .parsing import (
    extract_action,
    extract_action_input,
    extract_final,
    extract_thought,
    parse,
)
from .tools import Tool, ToolRegistry

__all__ = [
    "messages",
    "tools",
    "parsing",
    "memory",
    "agent",
    "ROLES",
    "Message",
    "Conversation",
    "Tool",
    "ToolRegistry",
    "extract_thought",
    "extract_final",
    "extract_action",
    "extract_action_input",
    "parse",
    "ConversationMemory",
    "Agent",
    "Step",
]
