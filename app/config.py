from typing import List

SILENCE_THRESHOLD_SECONDS = 10.0
DEFAULT_TARGET_TOPIC = "Global warming and climate change solutions"
DEFAULT_TARGET_DESCRIPTION = ""
DEFAULT_ALLOWED_LANGUAGE = "en"

# Allow all for prototype; tighten for production.
CORS_ALLOW_ORIGINS: List[str] = ["*"]
