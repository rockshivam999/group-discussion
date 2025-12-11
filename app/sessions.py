import logging
import random
import shlex
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import docker
except Exception:  # pragma: no cover - docker may be unavailable
    docker = None

from .config import (
    HISTORY_LIMIT,
    WLK_ARGS,
    WLK_HOST,
    WLK_IMAGE,
    WLK_MANAGED,
    WLK_PORT,
    WLK_SINGLETON,
)

logger = logging.getLogger(__name__)


def _pick_random_name(prefix: str) -> str:
    return f"{prefix}-{random.randint(1000, 9999)}"


@dataclass
class GroupSession:
    group_id: str
    topic: str
    description: str
    allowed_language: str
    target_embedding: Optional[object] = None
    wlk_container_id: Optional[str] = None
    wlk_ws_url: Optional[str] = None
    history: List[dict] = field(default_factory=list)
    last_summary_at: float = 0.0
    last_text_by_speaker: Dict[str, str] = field(default_factory=dict)
    last_digest_by_speaker: Dict[str, tuple] = field(default_factory=dict)
    lang_word_buffer_by_speaker: Dict[str, int] = field(default_factory=dict)
    last_lang_len_by_speaker: Dict[str, int] = field(default_factory=dict)
    last_profanity_len_by_speaker: Dict[str, int] = field(default_factory=dict)


class GroupRegistry:
    """Tracks per-group WhisperLiveKit endpoints and in-memory history."""

    def __init__(self) -> None:
        self.sessions: Dict[str, GroupSession] = {}
        self.max_history = HISTORY_LIMIT
        self.image = WLK_IMAGE
        self.external_host = WLK_HOST
        self.container_port = WLK_PORT
        self.singleton = WLK_SINGLETON
        self.managed = WLK_MANAGED
        self.shared_container_id: Optional[str] = None
        self.shared_ws_url: Optional[str] = None
        self.container_args = shlex.split(WLK_ARGS)
        if self.managed and docker:
            try:
                self.docker_client = docker.from_env()
            except Exception as exc:
                logger.warning("Docker client init failed: %s", exc)
                self.docker_client = None
        else:
            self.docker_client = None

    def ensure_session(
        self, group_id: str, topic: str, description: str, allowed_language: str, embedding
    ) -> GroupSession:
        session = self.sessions.get(group_id)
        if session:
            session.topic = topic
            session.description = description
            session.allowed_language = allowed_language
            session.target_embedding = embedding
            if not session.wlk_ws_url:
                cid, url = self._ensure_wlk_endpoint(group_id)
                session.wlk_container_id = cid
                session.wlk_ws_url = url
            # Reset per-speaker state for a clean restart
            session.last_text_by_speaker = {}
            session.last_lang_len_by_speaker = {}
            session.last_profanity_len_by_speaker = {}
            session.lang_word_buffer_by_speaker = {}
            return session

        container_id, ws_url = self._ensure_wlk_endpoint(group_id)
        session = GroupSession(
            group_id=group_id,
            topic=topic,
            description=description,
            allowed_language=allowed_language,
            target_embedding=embedding,
            wlk_container_id=container_id,
            wlk_ws_url=ws_url,
            last_lang_len_by_speaker={},
            last_profanity_len_by_speaker={},
        )
        self.sessions[group_id] = session
        return session

    def _ensure_wlk_endpoint(self, group_id: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.managed:
            url = f"ws://{self.external_host}:{self.container_port}/asr"
            if self.singleton:
                self.shared_ws_url = url
            return None, url

        if self.singleton:
            if self.shared_ws_url and self.shared_container_id:
                return self.shared_container_id, self.shared_ws_url
            cid, url = self._start_wlk_container("shared")
            self.shared_container_id = cid
            self.shared_ws_url = url
            return cid, url
        return self._start_wlk_container(group_id)

    def _start_wlk_container(self, name_hint: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.docker_client:
            logger.warning("Docker client not available; unable to start WhisperLiveKit container")
            return None, None

        container_name = _pick_random_name(f"wlk-{name_hint.lower()}")
        try:
            container = self.docker_client.containers.run(
                self.image,
                name=container_name,
                detach=True,
                ports={f"{self.container_port}/tcp": None},
                environment={"PORT": str(self.container_port)},
                command=self.container_args,
            )
            container.reload()
            host_port = container.attrs["NetworkSettings"]["Ports"][f"{self.container_port}/tcp"][0]["HostPort"]
            ws_url = f"ws://{self.external_host}:{host_port}/asr"
            logger.info("Started WhisperLiveKit for %s at %s (container=%s)", name_hint, ws_url, container.id)
            return container.id, ws_url
        except Exception as exc:  # pragma: no cover - runtime env may vary
            logger.error("Failed to start WhisperLiveKit container for %s: %s", name_hint, exc)
            return None, None

    def append_history(self, group_id: str, entry: dict) -> List[dict]:
        session = self.sessions.get(group_id)
        if not session:
            return []
        session.history.append(entry)
        if len(session.history) > self.max_history:
            session.history = session.history[-self.max_history :]
        return session.history

    def update_last(self, group_id: str, entry: dict) -> List[dict]:
        session = self.sessions.get(group_id)
        if not session or not session.history:
            return []
        session.history[-1] = entry
        return session.history

    def get_history(self, group_id: str) -> List[dict]:
        session = self.sessions.get(group_id)
        if not session:
            return []
        return session.history

    def get_pending_history(self, group_id: str, since_ts: float) -> List[dict]:
        session = self.sessions.get(group_id)
        if not session:
            return []
        return [e for e in session.history if e.get("timestamp", 0) > since_ts]

    def get_all_history(self, limit: int = 500) -> List[dict]:
        items: List[dict] = []
        for _, session in self.sessions.items():
            for entry in session.history:
                if entry:
                    items.append(entry)
        items.sort(key=lambda e: e.get("timestamp", 0))
        if limit and len(items) > limit:
            items = items[-limit:]
        return items

    def update_summary_timestamp(self, group_id: str) -> None:
        session = self.sessions.get(group_id)
        if session:
            session.last_summary_at = time.time()
