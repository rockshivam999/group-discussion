"""Application-wide configuration and defaults.

Environment variables can override these values where noted.
"""

import os
from typing import List


SILENCE_THRESHOLD_SECONDS = 10.0

# Default session metadata
DEFAULT_TARGET_TOPIC = "Classroom discussion"
DEFAULT_TARGET_DESCRIPTION = ""
DEFAULT_ALLOWED_LANGUAGE = "en"

# How long to merge consecutive snippets from the same speaker (seconds)
MERGE_WINDOW_SECONDS = 10.0

# Word-count thresholds
LANGUAGE_WORD_THRESHOLD = int(os.getenv("LANGUAGE_WORD_THRESHOLD", "10"))
PROFANITY_SNIPPET_LENGTH = 120

# History + summaries
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "200"))
LLM_SUMMARY_INTERVAL_SECONDS = float(os.getenv("LLM_SUMMARY_INTERVAL_SECONDS", "60"))
LLM_ENABLE_STUB = os.getenv("LLM_ENABLE_STUB", "1") not in {"0", "false", "False"}

# WhisperLiveKit container behavior
WLK_IMAGE = os.getenv("WLK_IMAGE", "quentinfuxa/whisperlivekit:latest")
WLK_HOST = os.getenv("WLK_HOST", "host.docker.internal")
WLK_PORT = int(os.getenv("WLK_PORT", "8000"))
WLK_SINGLETON = os.getenv("WLK_SINGLETON", "1") not in {"0", "false", "False"}
WLK_MANAGED = os.getenv("WLK_MANAGED", "1") not in {"0", "false", "False"}
WLK_ARGS = os.getenv("WLK_ARGS", "--diarization")

# Allow all for prototype; tighten for production.
CORS_ALLOW_ORIGINS: List[str] = ["*"]
