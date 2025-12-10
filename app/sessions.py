import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import shlex

try:
    import docker
except Exception:  # pragma: no cover - docker not always available in dev
    docker = None

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
    tap_task: Optional[object] = None  # asyncio.Task


class GroupRegistry:
    """
    Tracks per-group WhisperLiveKit containers and history.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, GroupSession] = {}
        self.max_history = int(os.getenv("HISTORY_LIMIT", "200"))
        self.image = os.getenv("WLK_IMAGE", "quentinfuxa/whisperlivekit:latest")
        # Use host.docker.internal by default so browsers on host can reach ephemeral ports
        self.external_host = os.getenv("WLK_HOST", "host.docker.internal")
        self.container_port = int(os.getenv("WLK_PORT", "8000"))
        self.singleton = os.getenv("WLK_SINGLETON", "1") not in ("0", "false", "False")
        self.managed = os.getenv("WLK_MANAGED", "1") not in ("0", "false", "False")
        self.shared_container_id: Optional[str] = None
        self.shared_ws_url: Optional[str] = None
        # Allow passing arbitrary WLK CLI args (e.g., "--diarization --model small")
        self.container_args = shlex.split(os.getenv("WLK_ARGS", "--diarization"))
        # Only initialize Docker client when managing containers
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
            # Update metadata if it changed
            session.topic = topic
            session.description = description
            session.allowed_language = allowed_language
            session.target_embedding = embedding
            if not session.wlk_ws_url:
                cid, url = self._ensure_wlk_endpoint(group_id)
                session.wlk_container_id = cid
                session.wlk_ws_url = url
                logger.info("Refreshed WhisperLiveKit for %s at %s", group_id, session.wlk_ws_url)
                print(f"[WLK] Refreshed for {group_id} -> {session.wlk_ws_url}")
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
        )
        self.sessions[group_id] = session
        return session

    def _ensure_wlk_endpoint(self, group_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        When singleton mode is on (default), share one WLK container across groups.
        Otherwise start a dedicated container per group.
        """
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
        """
        Launch a WhisperLiveKit container. If docker is unavailable, return (None, None)
        and rely on an externally provided WLK URL.
        """
        if not self.docker_client:
            logger.warning("Docker client not available; unable to start WhisperLiveKit container")
            print("[WLK] Docker client unavailable; cannot start container")
            return None, None

        container_name = _pick_random_name(f"wlk-{name_hint.lower()}")
        try:
            container = self.docker_client.containers.run(
                self.image,
                name=container_name,
                detach=True,
                ports={f"{self.container_port}/tcp": None},  # let Docker pick host port
                environment={"PORT": str(self.container_port)},
                command=self.container_args,
            )
            container.reload()
            host_port = container.attrs["NetworkSettings"]["Ports"][f"{self.container_port}/tcp"][0]["HostPort"]
            ws_url = f"ws://{self.external_host}:{host_port}/asr"
            logger.info("Started WhisperLiveKit for %s at %s (container=%s)", name_hint, ws_url, container.id)
            print(f"[WLK] Started for {name_hint} -> {ws_url} (container={container.id})")
            return container.id, ws_url
        except Exception as exc:  # pragma: no cover - runtime env may vary
            logger.error("Failed to start WhisperLiveKit container for %s: %s", name_hint, exc)
            print(f"[WLK] Failed to start for {name_hint}: {exc}")
            return None, None

    def append_history(self, group_id: str, entry: dict) -> List[dict]:
        session = self.sessions.get(group_id)
        if not session:
            return []
        session.history.append(entry)
        if len(session.history) > self.max_history:
            session.history = session.history[-self.max_history :]
        return session.history

    def get_history(self, group_id: str) -> List[dict]:
        session = self.sessions.get(group_id)
        if not session:
            return []
        return session.history

    def update_summary_timestamp(self, group_id: str) -> None:
        session = self.sessions.get(group_id)
        if session:
            session.last_summary_at = time.time()
