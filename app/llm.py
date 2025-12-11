import logging
import random
from collections import Counter
from typing import Dict, List

from .analysis import compute_dominance

logger = logging.getLogger(__name__)


async def summarize_history(group_id: str, history: List[dict], topic: str, description: str) -> Dict:
    """
    Stub for a downstream LLM call.

    Replace this with a real API call; keep signature stable for drop-in.
    """
    # Simple heuristics until an external model is wired up
    dominance, dominant_speaker = compute_dominance(history)
    speakers = [h.get("speaker") or "unknown" for h in history if h.get("speaker")]
    counts = Counter(speakers)
    common_speakers = ", ".join(f"{s}:{c}" for s, c in counts.most_common(3)) if counts else ""

    alerts = []
    if random.random() > 0.6:
        alerts.append({"type": "TOPIC_LLM", "msg": "Possible off-topic drift (stub)"})
    if random.random() > 0.7:
        alerts.append({"type": "DOMINANCE", "msg": f"{dominant_speaker or 'someone'} may be dominating"})

    summary_text = (
        f"Summary for {group_id}: topic='{topic}'. Recent speakers: {common_speakers}. "
        f"Notes: heuristic stub; replace with real LLM output."
    )

    result = {
        "summary": summary_text,
        "alerts": alerts,
        "dominance": dominance,
        "dominant_speaker": dominant_speaker,
        "target_topic": topic,
        "target_description": description,
    }
    logger.info("LLM stub for %s result=%s", group_id, result)
    return result
