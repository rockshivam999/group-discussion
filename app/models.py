from typing import Optional

from pydantic import BaseModel

from .config import DEFAULT_ALLOWED_LANGUAGE, DEFAULT_TARGET_DESCRIPTION, DEFAULT_TARGET_TOPIC


class SessionRequest(BaseModel):
    topic: str = DEFAULT_TARGET_TOPIC
    description: str = DEFAULT_TARGET_DESCRIPTION
    allowed_language: str = DEFAULT_ALLOWED_LANGUAGE
    # If provided, backend skips container launch and uses this URL directly
    wlk_ws_url: Optional[str] = None


class SessionResponse(BaseModel):
    group_id: str
    wlk_ws_url: Optional[str]
    analysis_ws_url: str
    allowed_language: str
    topic: str
    description: str


class IngestEvent(BaseModel):
    text: str
    lang: Optional[str] = None
    speaker: Optional[object] = None  # accept int/str
    timestamp: Optional[float] = None
    source: str = "wlk"
    raw_payload: Optional[dict] = None
