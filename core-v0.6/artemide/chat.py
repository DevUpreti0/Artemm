"""Simple, explicit chat serialization for the base causal language model."""
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

_ALLOWED_ROLES = {"system", "user", "assistant"}

def format_chat_prompt(messages: Iterable[ChatMessage], add_assistant_prefix=True) -> str:
    """Serialize messages without pretending the model has instruction tuning."""
    parts = []
    for message in messages:
        role = message.role.strip().lower()
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Unsupported chat role: {message.role}")
        content = message.content.strip()
        if not content:
            raise ValueError("Chat message content cannot be empty")
        parts.append(f"<|{role}|>\n{content}\n")
    if not parts:
        raise ValueError("At least one chat message is required")
    if add_assistant_prefix:
        parts.append("<|assistant|>\n")
    return "".join(parts)
