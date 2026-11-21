"""Tool registry for the lightweight agent framework."""

from typing import Callable, Dict, List, Optional


class Tool:
    """A named callable with a description the agent can reason about."""

    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, tool_input: str = ""):
        return self.func(tool_input)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class ToolRegistry:
    """A registry of named :class:`Tool` objects."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    @property
    def tools(self) -> List[Tool]:
        return list(self._tools.values())

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError("register expects a Tool instance")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def add(self, name: str, func: Callable, description: str = "") -> Tool:
        """Register ``func`` under ``name`` and return the :class:`Tool`."""
        return self.register(Tool(name, func, description))

    def tool(self, name: Optional[str] = None, description: str = ""):
        """Decorator: register a function as a tool."""

        def decorator(func: Callable) -> Callable:
            self.register(Tool(name or func.__name__, func, description))
            return func

        return decorator

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"no tool named {name!r}")
        return self._tools[name]

    def call(self, name: str, tool_input: str = ""):
        return self.get(name)(tool_input)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
