import logging
from typing import Tuple

from better_profanity import profanity
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel

from .config import DEFAULT_TARGET_DESCRIPTION, DEFAULT_TARGET_TOPIC

logger = logging.getLogger(__name__)

# Model instances are loaded once per process
ASR_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
TOPIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
profanity.load_censor_words()


def encode_topic(topic: str, description: str = ""):
    text = f"{topic}. {description}".strip()
    return text, TOPIC_MODEL.encode(text, convert_to_tensor=True)


def analyze_text(
    text: str,
    language: str,
    target_embedding,
    target_topic: str,
    similarity_floor: float = 0.15,
):
    alerts = []
    similarity = 0.0

    if profanity.contains_profanity(text):
        alerts.append({"type": "OFFENSIVE", "msg": "Foul language detected"})

    if language and language != "en":
        alerts.append({"type": "LANGUAGE", "msg": f"Speaking {language} instead of English"})

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


def compute_dominance(speech_duration: float, chunk_duration: float) -> str:
    if chunk_duration <= 0:
        return "QUIET"
    ratio = speech_duration / chunk_duration
    if ratio > 0.75:
        return "DOMINATING"
    if ratio > 0.35:
        return "BALANCED"
    return "QUIET"
