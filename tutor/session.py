"""Practice session management."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tutor.scenarios import Scenario


@dataclass
class Message:
    role: str  # "student" or "tutor"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    scenario: Scenario
    messages: List[Message] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    feedback: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def end(self) -> None:
        self.ended_at = time.time()

    def get_conversation_log(self) -> str:
        """Format conversation for feedback analysis."""
        lines = []
        for msg in self.messages:
            label = "Student" if msg.role == "student" else f"Tutor ({self.scenario.name})"
            lines.append(f"{label}: {msg.content}")
        return "\n\n".join(lines)

    def to_claude_messages(self) -> List[dict]:
        """Convert to Claude API message format."""
        msgs = []
        for msg in self.messages:
            role = "user" if msg.role == "student" else "assistant"
            msgs.append({"role": role, "content": msg.content})
        return msgs


class SessionManager:
    """Manages active practice sessions per user."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def start_session(self, user_id: str, scenario: Scenario) -> Session:
        """Start a new practice session."""
        import uuid
        session = Session(
            session_id=uuid.uuid4().hex[:12],
            scenario=scenario,
        )
        self._sessions[user_id] = session
        return session

    def get_session(self, user_id: str) -> Optional[Session]:
        """Get active session for user."""
        session = self._sessions.get(user_id)
        if session and session.is_active:
            return session
        return None

    def end_session(self, user_id: str) -> Optional[Session]:
        """End and return the session."""
        session = self._sessions.get(user_id)
        if session and session.is_active:
            session.end()
            return session
        return None

    def has_active_session(self, user_id: str) -> bool:
        return self.get_session(user_id) is not None
