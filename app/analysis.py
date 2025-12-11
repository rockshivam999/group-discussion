import logging
import re
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

from better_profanity import profanity

logger = logging.getLogger(__name__)

profanity.load_censor_words()


def collapse_repetitions(text: str) -> str:
    """Heuristic cleanup to reduce runaway repetitions from streaming ASR."""
    if not text:
        return text

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
        if streak < 2:
            collapsed.append(tok)
    cleaned = " ".join(collapsed)

    cleaned = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1 \1", cleaned, flags=re.UNICODE | re.IGNORECASE)
    cleaned = re.sub(r"((?:\S+\s+){2,7}\S+)(?:\s+\1){1,}", r"\1", cleaned, flags=re.UNICODE | re.IGNORECASE)

    if len(cleaned) > 600:
        cleaned = cleaned[:600].rstrip() + " ..."
    return cleaned.strip()


def diff_new_segment(previous: str, current: str) -> str:
    """Return only the newly added segment when current text extends previous."""
    if not current:
        return ""
    if previous and current.startswith(previous):
        return current[len(previous) :].strip()
    return current.strip()


def profanity_alerts(text: str, speaker: str, lang: str, max_len: int = 120) -> list:
    if not text:
        return []
    if profanity.contains_profanity(text):
        snippet = text.strip()[:max_len]
        return [
            {
                "type": "OFFENSIVE",
                "msg": f"Foul or inappropriate language detected: {snippet} (speaker={speaker}, lang={lang})",
            }
        ]
    return []


def language_alerts(text: str, lang: str, allowed_language: str) -> list:
    lang_normalized = (lang or "").lower()
    allowed = (allowed_language or "").lower()
    if allowed and lang_normalized and lang_normalized != allowed:
        snippet = text.strip()[:120]
        return [
            {
                "type": "LANGUAGE",
                "msg": f"Speaking {lang_normalized} instead of {allowed}: {snippet}",
            }
        ]
    return []


def compute_dominance(history: Iterable[dict], window: int = 30) -> Tuple[str, Optional[str]]:
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
