"""Message and conversation primitives for the lightweight agent framework."""

from dataclasses import dataclass, field
from typing import Iterator, List

ROLES = ("system", "user", "assistant", "tool", "observation")


@dataclass
class Message:
    """A single chat message."""

    role: str
    content: str

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {ROLES}")

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """An ordered list of messages."""

    messages: List[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> Message:
        msg = Message(role, content)
        self.messages.append(msg)
        return msg

    def by_role(self, role: str) -> List[Message]:
        return [m for m in self.messages if m.role == role]

    def clear(self) -> None:
        self.messages.clear()

    def to_list(self) -> List[dict]:
        return [m.to_dict() for m in self.messages]

    def to_prompt(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)
