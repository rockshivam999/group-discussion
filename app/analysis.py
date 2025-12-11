import logging
import re
from collections import Counter
from typing import Iterable, Optional, Tuple

from better_profanity import profanity
from sentence_transformers import SentenceTransformer, util

from .config import DEFAULT_ALLOWED_LANGUAGE

logger = logging.getLogger(__name__)

# Model instances are loaded once per process
TOPIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
profanity.load_censor_words()


def encode_topic(topic: str, description: str = ""):
    text = f"{topic}. {description}".strip()
    return text, TOPIC_MODEL.encode(text, convert_to_tensor=True)


def analyze_live_alerts(
    text: str,
    language: str,
    allowed_language: str = DEFAULT_ALLOWED_LANGUAGE,
    check_language: bool = True,
    check_profanity: bool = True,
):
    alerts = []
    lang = (language or "").lower()
    allowed = (allowed_language or "").lower()

    if check_profanity and profanity.contains_profanity(text):
        snippet = text.strip()[:80]
        alerts.append({"type": "OFFENSIVE", "msg": f"Foul or inappropriate language detected: {snippet}"})

    if check_language and allowed and lang and lang != allowed:
        alerts.append({"type": "LANGUAGE", "msg": f"Speaking {lang} instead of {allowed}"})

    return alerts


def analyze_text(
    text: str,
    language: str,
    target_embedding,
    target_topic: str,
    allowed_language: str = DEFAULT_ALLOWED_LANGUAGE,
    similarity_floor: float = 0.15,
):
    alerts = []
    similarity = 0.0
    lang = (language or "").lower()
    allowed = (allowed_language or "").lower()

    if profanity.contains_profanity(text):
        alerts.append({"type": "OFFENSIVE", "msg": "Foul or inappropriate language detected"})

    if allowed and lang and lang != allowed:
        alerts.append({"type": "LANGUAGE", "msg": f"Speaking {lang} instead of {allowed}"})

    if target_topic and target_embedding is not None and text.strip():
        text_emb = TOPIC_MODEL.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_emb, target_embedding).item()
        if similarity < similarity_floor:
            alerts.append(
                {
                    "type": "TOPIC",
                    "msg": "Drifting off-topic",
                    "score": f"{similarity:.2f}",
                }
            )

    return alerts, similarity


def collapse_repetitions(text: str) -> str:
    """
    Heuristic cleanup to reduce runaway repetitions from streaming ASR.
    - Collapse same token repeated 3+ times in a row down to 2.
    - Collapse short phrases (3-8 tokens) repeated back-to-back (Unicode-safe).
    - Soft cap overly long text to keep UI/LLM stable.
    """
    if not text:
        return text

    # Token-level collapse (language agnostic)
    tokens = text.split()
    collapsed = []
    last = None
    streak = 0
    for tok in tokens:
        if tok == last:
            streak += 1
        else:
            streak = 0
            last = tok
        if streak < 2:  # allow at most 2 in a row
            collapsed.append(tok)
    cleaned = " ".join(collapsed)

    # Phrase-level collapse (use \S to include non-Latin scripts)
    cleaned = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1 \1", cleaned, flags=re.UNICODE | re.IGNORECASE)
    cleaned = re.sub(r"((?:\S+\s+){2,7}\S+)(?:\s+\1){1,}", r"\1", cleaned, flags=re.UNICODE | re.IGNORECASE)

    # Soft cap length for safety
    if len(cleaned) > 600:
        cleaned = cleaned[:600].rstrip() + " ..."
    return cleaned.strip()


def compute_dominance(history: Iterable[dict], window: int = 30) -> Tuple[str, Optional[str]]:
    """
    Approximate participation balance from recent entries.

    - If a single speaker holds >=70% of turns, mark DOMINATING and return that speaker.
    - If entries exist but no dominant speaker, mark BALANCED.
    - If little/no activity, mark QUIET.
    """
    recent = [h for h in history if h.get("text")][-window:]
    if not recent:
        return "QUIET", None

    counts = Counter(entry.get("speaker") or "unknown" for entry in recent)
    total = sum(counts.values())
    speaker, turns = counts.most_common(1)[0]
    ratio = turns / total if total else 0

    if ratio >= 0.7:
        return "DOMINATING", speaker
    if ratio >= 0.3:
        return "BALANCED", None
    return "QUIET", None
