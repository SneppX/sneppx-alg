"""Conversation memory strategies for the lightweight agent framework."""

from typing import List, Optional

from .messages import Conversation, Message


class ConversationMemory:
    """Keeps the last ``max_turns`` user/assistant exchanges plus any system text."""

    def __init__(self, system: Optional[str] = None, max_turns: Optional[int] = None):
        self.system = system
        self.max_turns = max_turns
        self.conversation = Conversation()
        if system:
            self.conversation.add("system", system)

    def add(self, role: str, content: str) -> Message:
        msg = self.conversation.add(role, content)
        self._trim()
        return msg

    def _trim(self) -> None:
        if self.max_turns is None:
            return
        system = [m for m in self.conversation if m.role == "system"]
        others = [m for m in self.conversation if m.role != "system"]
        keep = self.max_turns * 2
        if len(others) > keep:
            shortened = system + others[-keep:]
            self.conversation = Conversation(shortened)

    def messages(self) -> List[Message]:
        return list(self.conversation)

    def to_list(self) -> List[dict]:
        return self.conversation.to_list()

    def clear(self) -> None:
        self.conversation.clear()
        if self.system:
            self.conversation.add("system", self.system)

    def __len__(self) -> int:
        return len(self.conversation)
