import logging
import time
from typing import Dict, Tuple

from .analysis import collapse_repetitions, diff_new_segment, language_alerts, profanity_alerts
from .config import LANGUAGE_WORD_THRESHOLD, MERGE_WINDOW_SECONDS, PROFANITY_SNIPPET_LENGTH
from .connections import ConnectionManager
from .sessions import GroupRegistry, GroupSession

logger = logging.getLogger(__name__)


class EventProcessor:
    """Ingest transcript snippets, run checks, and fan out alerts."""

    def __init__(
        self,
        registry: GroupRegistry,
        manager: ConnectionManager,
        language_word_threshold: int = LANGUAGE_WORD_THRESHOLD,
    ) -> None:
        self.registry = registry
        self.manager = manager
        self.language_word_threshold = language_word_threshold

    async def process_event(self, group_id: str, payload, session: GroupSession) -> Dict:
        ts = payload.timestamp or time.time()
        speaker_val = payload.speaker
        speaker = str(speaker_val) if speaker_val is not None else "unknown"
        lang = payload.lang or "unknown"

        # Per-speaker short-circuit: drop identical text+lang seen very recently to prevent floods after silence
        dedupe_key = f"{speaker}|{lang}"
        last_digest = session.last_text_by_speaker.get(dedupe_key)

        base_entry: Dict = {
            "group_id": group_id,
            "source": payload.source or "wlk",
            "speaker": speaker,
            "text": collapse_repetitions(payload.text or ""),
            "lang": lang,
            "timestamp": ts,
            "target_topic": session.topic,
            "target_description": session.description,
            "allowed_language": session.allowed_language,
        }

        alerts = []
        prev_text = session.last_text_by_speaker.get(speaker, "")
        words = base_entry["text"].split()
        total_words = len(words)

        # Profanity: check only newly added words since last profanity scan for this speaker
        prev_prof_idx = session.last_profanity_len_by_speaker.get(speaker, 0)
        if total_words > prev_prof_idx:
            prof_segment = " ".join(words[prev_prof_idx:])
            alerts.extend(profanity_alerts(prof_segment, speaker, lang, max_len=PROFANITY_SNIPPET_LENGTH))
            session.last_profanity_len_by_speaker[speaker] = total_words

        # Language: accumulate newly added words since last language scan
        prev_lang_len = session.last_lang_len_by_speaker.get(speaker, 0)
        new_lang_words = max(0, total_words - prev_lang_len)
        new_words_count = new_lang_words
        if new_lang_words > 0:
            session.last_lang_len_by_speaker[speaker] = total_words
            buf = session.lang_word_buffer_by_speaker.get(speaker, 0) + new_lang_words
            while buf >= self.language_word_threshold:
                lang_segment = " ".join(words[-self.language_word_threshold :]) if words else base_entry["text"]
                alerts.extend(language_alerts(lang_segment, lang, session.allowed_language))
                buf -= self.language_word_threshold
            session.lang_word_buffer_by_speaker[speaker] = buf

        # Use the new_lang_words as the growth metric for merge guards

        # Early guard: drop exact same text for same speaker/lang and dedupe within 1 second
        digest = (base_entry["text"], base_entry["lang"])
        last_digest = session.last_digest_by_speaker.get(speaker)
        if last_digest:
            last_text, last_lang, last_ts = last_digest
            if last_text == digest[0] and last_lang == digest[1] and (ts - last_ts) < 3.0:
                history = self.registry.get_history(group_id)
                return history[-1] if history else base_entry

        session.last_digest_by_speaker[speaker] = (digest[0], digest[1], ts)

        session.last_text_by_speaker[speaker] = base_entry["text"]

        history = self.registry.get_history(group_id)
        previous_entry = history[-1] if history else None
        # Keep timestamps monotonically increasing per group to avoid re-merging old chunks
        if history:
            last_ts = history[-1].get("timestamp", 0)
            if ts <= last_ts:
                ts = last_ts + 0.001
                base_entry["timestamp"] = ts
        if history:
            last = history[-1]
            if (last.get("speaker") or "unknown") == speaker and (last.get("text") or "") == base_entry["text"]:
                return last
            same_speaker = (last.get("speaker") or "unknown") == speaker
            close_in_time = ts - (last.get("timestamp") or 0) < MERGE_WINDOW_SECONDS
            small_change = new_words_count < 3 and (len(base_entry["text"]) - len(last.get("text") or "")) < 4
            if (
                same_speaker
                and close_in_time
                and base_entry["text"]
                and (base_entry["text"] in (last.get("text") or ""))
                and len(base_entry["text"]) <= len(last.get("text") or "")
            ):
                return last
            if same_speaker and small_change:
                return last

        merged = False
        if history:
            last = history[-1]
            same_speaker = (last.get("speaker") or "unknown") == speaker
            close_in_time = ts - (last.get("timestamp") or 0) < MERGE_WINDOW_SECONDS
            if same_speaker and close_in_time:
                prev_text = (last.get("text") or "").strip()
                curr_text = (base_entry.get("text") or "").strip()
                if curr_text.startswith(prev_text):
                    merged_text = curr_text
                elif prev_text.startswith(curr_text):
                    merged_text = prev_text
                else:
                    merged_text = f"{prev_text} {curr_text}".strip()
                base_entry["text"] = collapse_repetitions(merged_text)
                merged = True
                history[-1] = base_entry

        dominance_state, dominance_speaker = ("PENDING_LLM", None)
        entry = {
            **base_entry,
            "alerts": alerts,
            "topic_score": "",
            "dominance_state": dominance_state,
            "dominance_speaker": dominance_speaker,
            "speech_ratio": None,
            "silence": False,
            "chunk_seconds": None,
        }

        if merged:
            self.registry.update_last(group_id, entry)
        else:
            self.registry.append_history(group_id, entry)

        # Drop duplicate broadcasts if nothing meaningful changed
        if previous_entry:
            same_text = entry["text"] == previous_entry.get("text")
            same_lang = entry.get("lang") == previous_entry.get("lang")
            same_alerts = entry.get("alerts") == previous_entry.get("alerts")
            if same_text and same_lang and same_alerts:
                return previous_entry

        await self.manager.broadcast_alert(entry)
        logger.info("Ingested event for %s (merged=%s): %s", group_id, merged, entry)
        return entry


def build_entry_from_summary(group_id: str, summary: Dict) -> Dict:
    return {
        "group_id": group_id,
        "source": "llm",
        "text": summary.get("summary", "LLM summary"),
        "alerts": summary.get("alerts", []),
        "timestamp": time.time(),
        "dominance_state": summary.get("dominance", "PENDING"),
        "dominance_speaker": summary.get("dominant_speaker"),
        "topic_score": "",
        "lang": "unknown",
        "target_topic": summary.get("target_topic"),
        "target_description": summary.get("target_description"),
        "speaker": summary.get("dominant_speaker"),
        "speech_ratio": summary.get("speech_ratio"),
        "silence": False,
        "chunk_seconds": None,
    }
