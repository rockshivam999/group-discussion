import { useRef, useState } from "react";
import TranscriptList from "../components/TranscriptList";
import { apiBase, wsBase } from "../lib/config";
import { useNavigate } from "react-router-dom";

const LANG_OPTIONS = [
  { code: "en", label: "English" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "hi", label: "Hindi" },
];

const WORD_THRESHOLD = 10;
const SILENCE_SPLIT_SECONDS = 10;
const DEDUPE_WINDOW_SECONDS = 8;

export default function StudentPage() {
  const [groupId, setGroupId] = useState("Group-Alpha");
  const [topic, setTopic] = useState("Nature");
  const [description, setDescription] = useState("Discuss biodiversity and why protecting nature matters.");
  const [allowedLanguage, setAllowedLanguage] = useState("en");
  const [isRecording, setIsRecording] = useState(false);
  const [studentStatus, setStudentStatus] = useState("Idle");
  const [analysisStatus, setAnalysisStatus] = useState("Idle");
  const mediaRecorderRef = useRef(null);
  const wlkSocketRef = useRef(null);
  const analysisSocketRef = useRef(null);
  const [wlkTranscripts, setWlkTranscripts] = useState([]);
  const [flags, setFlags] = useState([]);
  const [analysisTranscripts, setAnalysisTranscripts] = useState([]);
  const speakerBuffersRef = useRef({});
  const recentTextRef = useRef({});
  const lastFrameSignatureRef = useRef(null);
  const lastSentRef = useRef({});
  const lastSignatureRef = useRef({});
  const lastEndRef = useRef({});
  const lastFullTextRef = useRef({});
  const sentEventsRef = useRef({});
  const navigate = useNavigate();

  const cleanText = (t) => {
    if (!t) return "";
    let cleaned = t.replace(/\b(\w+)(?:\s+\1\b){2,}/gi, "$1 $1");
    cleaned = cleaned.replace(/(\b[\w'-]+(?:\s+[\w'-]+){2,7})\s+(?:\1\s*){1,}/gi, "$1");
    return cleaned.trim();
  };

  const startRecording = async () => {
    const gid = groupId.trim();
    if (!gid || !topic.trim()) {
      alert("Please enter Group ID and Topic before starting.");
      return;
    }

    setStudentStatus("Starting session...");
    try {
      const res = await fetch(`${apiBase}/groups/${encodeURIComponent(gid)}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topic.trim(),
          description: description.trim(),
          allowed_language: allowedLanguage,
        }),
      });
      if (!res.ok) {
        setStudentStatus(`Session error: ${await res.text()}`);
        return;
      }
      const session = await res.json();
      if (!session.wlk_ws_url) {
        setStudentStatus("WhisperLiveKit URL unavailable. Check backend logs.");
        return;
      }

      // Prefill with existing history so previous runs are visible
      fetch(`${apiBase}/groups/${encodeURIComponent(gid)}/history`)
        .then((res) => res.json())
        .then((json) => {
          const hist = (json.history || []).slice().reverse();
          setAnalysisTranscripts(hist.map((h) => ({ ...h, speaker: h.speaker || h.dominance_speaker })));
          const flagged = hist.filter((h) => (h.alerts || []).length > 0 || h.source === "llm");
          setFlags(flagged);
        })
        .catch((err) => console.error("history fetch failed", err));

      const analysisSocket = new WebSocket(`${wsBase}/ws/group/${encodeURIComponent(gid)}`);
      analysisSocketRef.current = analysisSocket;
      analysisSocket.onopen = () => setAnalysisStatus("Connected");
      analysisSocket.onclose = () => setAnalysisStatus("Disconnected");
      analysisSocket.onerror = () => setAnalysisStatus("Error");
      analysisSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const entryAlerts = data.alerts || [];
        const entry = {
          text: data.text,
          timestamp: data.timestamp,
          lang: data.lang,
          topic_score: data.topic_score,
          alerts: entryAlerts,
          dominance: data.dominance_state,
          speech_ratio: data.speech_ratio,
          silence: data.silence,
          target_topic: data.target_topic,
          source: data.source || "analysis",
          speaker: data.dominance_speaker || data.speaker,
        };
        setAnalysisTranscripts((prev) => [entry, ...prev].slice(0, 200));
        if (entryAlerts.length > 0 || data.source === "llm") {
          setFlags((prev) => [entry, ...prev].slice(0, 120));
        }
      };

      const wlkSocket = new WebSocket(session.wlk_ws_url);
      wlkSocketRef.current = wlkSocket;
      setStudentStatus("Connecting to WhisperLiveKit...");
      lastFrameSignatureRef.current = null;

      wlkSocket.onopen = async () => {
        setStudentStatus("Streaming to WhisperLiveKit...");
        sentEventsRef.current = {};
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const preferredMime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm";
        const recorder = new MediaRecorder(stream, { mimeType: preferredMime });
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0 && wlkSocket.readyState === WebSocket.OPEN) {
            wlkSocket.send(event.data);
          }
        };
        recorder.onerror = (err) => {
          console.error("MediaRecorder error", err);
          setStudentStatus("Recorder error");
          stopRecording();
        };
        recorder.start(500);
        setIsRecording(true);
      };

      wlkSocket.onmessage = (event) => {
        let payload = {};
        try {
          payload = JSON.parse(event.data);
        } catch (_) {
          payload = { text: event.data };
        }

        // Drop frames that are byte-for-byte identical to the previous one (matches WLK sample UI behavior)
        const frameSignature = JSON.stringify(
          (payload.lines || []).map((l) => ({
            s: l.speaker,
            t: l.text,
            st: l.start,
            e: l.end,
            lang: l.detected_language || payload.lang || payload.language,
          }))
        );
        if (lastFrameSignatureRef.current === frameSignature) {
          return;
        }
        lastFrameSignatureRef.current = frameSignature;

        // Use only the most recent non-empty, non-silence line by end time
        const candidates = (payload.lines || []).filter((l) => l.text && l.text.trim() && l.speaker !== -2);
        if (candidates.length === 0) return;

        const parseEndToSeconds = (endStr) => {
          if (!endStr || typeof endStr !== "string") return null;
          const parts = endStr.split(":").map(Number);
          if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
          if (parts.length === 2) return parts[0] * 60 + parts[1];
          return null;
        };

        const latest = candidates.reduce((acc, cur) => {
          const accEnd = parseEndToSeconds(acc?.end) || 0;
          const curEnd = parseEndToSeconds(cur?.end) || 0;
          return curEnd > accEnd ? cur : acc;
        }, candidates[0]);

        const endSeconds = parseEndToSeconds(latest.end);

        const entry = {
          text: cleanText(latest.text || ""),
          speaker: latest.speaker ?? "unknown",
          lang: latest.detected_language || payload.lang || payload.language || "unknown",
          timestamp: endSeconds || Date.now() / 1000,
          source: "wlk",
        };

        // Skip if this line end timestamp has not advanced enough (avoid replays after silence)
        const lastEnd = lastEndRef.current[entry.speaker] || 0;
        const silenceGap = endSeconds && lastEnd && endSeconds - lastEnd > SILENCE_SPLIT_SECONDS;
        if (endSeconds && endSeconds <= lastEnd + 0.25) {
            return;
        }
        if (endSeconds) {
            lastEndRef.current[entry.speaker] = endSeconds;
        }

        const key = `${entry.speaker}|${entry.lang}`;
        const signature = `${key}|${entry.text}`;
        const lastSig = lastSignatureRef.current[key];
        if (lastSig && lastSig.signature === signature && entry.timestamp - lastSig.ts < 3) {
          return; // drop exact replays within cooldown
        }
        lastSignatureRef.current[key] = { signature, ts: entry.timestamp };

        setWlkTranscripts((prev) => [entry, ...prev].slice(0, 200));

        // Drop loops where WLK replays the same text variants for the same speaker in quick succession
        const recent = recentTextRef.current[entry.speaker] || [];
        const now = entry.timestamp;
        const fresh = recent.filter((r) => now - r.ts < DEDUPE_WINDOW_SECONDS);
        const seenRecently = fresh.some((r) => r.text === entry.text);
        if (seenRecently) {
          recentTextRef.current[entry.speaker] = fresh;
          lastFullTextRef.current[entry.speaker] = entry.text;
          return;
        }
        fresh.push({ text: entry.text, ts: now });
        recentTextRef.current[entry.speaker] = fresh.slice(-5);

        // Compute only the newly added segment for this speaker (front-end diff)
        const prevFull = lastFullTextRef.current[entry.speaker] || "";
        const isExtension = !!prevFull && entry.text.startsWith(prevFull);
        const newSegment = isExtension ? entry.text.slice(prevFull.length).trim() : entry.text;
        const words = newSegment.split(/\s+/).filter(Boolean);
        if (words.length === 0) {
          lastFullTextRef.current[entry.speaker] = entry.text;
          return;
        }

        // After a long silence, treat the next chunk as baseline without sending to backend
        if (silenceGap) {
          speakerBuffersRef.current[entry.speaker] = { wordCount: words.length, ts: entry.timestamp, text: entry.text };
          lastFullTextRef.current[entry.speaker] = entry.text;
          recentTextRef.current[entry.speaker] = [{ text: entry.text, ts: entry.timestamp }];
          lastSignatureRef.current[key] = { signature, ts: entry.timestamp };
          return;
        }

        // Never resend text with a timestamp that is not advancing for this speaker
        const lastSent = lastSentRef.current[entry.speaker];
        if (lastSent && entry.timestamp <= lastSent.ts) {
          speakerBuffersRef.current[entry.speaker] = { wordCount: words.length, ts: entry.timestamp, text: entry.text };
          lastFullTextRef.current[entry.speaker] = entry.text;
          recentTextRef.current[entry.speaker] = fresh.slice(-5);
          return;
        }

        const buffers = speakerBuffersRef.current;
        const buf = buffers[entry.speaker] || { wordCount: 0, ts: 0, text: "" };
        const wordCount = (buf.wordCount || 0) + words.length;
        const timeGap = entry.timestamp - (buf.ts || 0);
        const charGrowth = newSegment.length;

        const shouldSend =
          !buf.text ||
          timeGap > SILENCE_SPLIT_SECONDS ||
          (isExtension && (words.length >= WORD_THRESHOLD || charGrowth >= 15));

        if (shouldSend) {
          ingestToBackend({
            text: newSegment,
            lang: entry.lang,
            speaker: entry.speaker,
            timestamp: entry.timestamp,
            source: entry.source,
          });
          lastSentRef.current[entry.speaker] = { len: newSegment.length, ts: entry.timestamp };
        }
        buffers[entry.speaker] = { wordCount, ts: entry.timestamp, text: entry.text };
        lastFullTextRef.current[entry.speaker] = entry.text;
      };

      wlkSocket.onerror = () => setStudentStatus("WhisperLiveKit error");
      wlkSocket.onclose = () => stopRecording();
    } catch (err) {
      console.error(err);
      setStudentStatus("Failed to start session");
    }
  };

  const ingestToBackend = async (entry, rawPayload = null) => {
    const gid = groupId.trim();
    if (!gid) return;

    // Drop repeats for the same speaker/lang/text within a short window to avoid backend floods on replayed frames
    const dedupeKey = `${entry.speaker}|${entry.lang}|${entry.text}|${Math.round(entry.timestamp || 0)}`;
    const lastSeen = sentEventsRef.current[dedupeKey];
    if (lastSeen && entry.timestamp && entry.timestamp - lastSeen < 30) {
      return;
    }
    sentEventsRef.current[dedupeKey] = entry.timestamp || Date.now() / 1000;

    try {
      await fetch(`${apiBase}/groups/${encodeURIComponent(gid)}/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: entry.text,
          lang: entry.lang,
          speaker: entry.speaker != null ? String(entry.speaker) : "unknown",
          timestamp: entry.timestamp,
          source: "wlk",
          raw_payload: rawPayload,
        }),
      });
    } catch (err) {
      console.error("Failed to ingest event", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    }
    mediaRecorderRef.current = null;

    if (wlkSocketRef.current) {
      try {
        wlkSocketRef.current.close();
      } catch (err) {
        console.error(err);
      }
      wlkSocketRef.current = null;
    }

    if (analysisSocketRef.current) {
      try {
        analysisSocketRef.current.close();
      } catch (err) {
        console.error(err);
      }
      analysisSocketRef.current = null;
    }

    speakerBuffersRef.current = {};
    lastFrameSignatureRef.current = null;
    sentEventsRef.current = {};
    setIsRecording(false);
    setStudentStatus("Stopped");
    setAnalysisStatus("Idle");
  };

  return (
    <div className="grid grid-2">
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Student Mic</h2>
        <label className="muted">Group ID</label>
        <input
          value={groupId}
          onChange={(e) => setGroupId(e.target.value)}
          placeholder="e.g., Group-A"
          style={{
            width: "100%",
            padding: "12px",
            borderRadius: "10px",
            border: "1px solid #e2e8f0",
            margin: "6px 0 12px 0",
          }}
        />
        <label className="muted">Topic</label>
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g., Climate change solutions"
          style={{
            width: "100%",
            padding: "12px",
            borderRadius: "10px",
            border: "1px solid #e2e8f0",
            margin: "6px 0 12px 0",
          }}
        />
        <label className="muted">Short Description (for off-topic check)</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Key points or question students should discuss"
          rows={3}
          style={{
            width: "100%",
            padding: "12px",
            borderRadius: "10px",
            border: "1px solid #e2e8f0",
            margin: "6px 0 12px 0",
            resize: "vertical",
          }}
        />

        <label className="muted">Allowed Language</label>
        <select
          value={allowedLanguage}
          onChange={(e) => setAllowedLanguage(e.target.value)}
          style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #e2e8f0", marginBottom: "12px" }}
        >
          {LANG_OPTIONS.map((opt) => (
            <option key={opt.code} value={opt.code}>
              {opt.label}
            </option>
          ))}
        </select>

        <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
          {!isRecording ? (
            <button className="button primary" onClick={startRecording}>
              Start Recording
            </button>
          ) : (
            <button className="button danger" onClick={stopRecording}>
              Stop
            </button>
          )}
          <span className="pill" style={{ background: "#f1f5f9" }}>WLK: {studentStatus}</span>
          <span className="pill" style={{ background: "#e0f2fe", color: "#075985" }}>Backend: {analysisStatus}</span>
        </div>

        <p className="muted" style={{ marginTop: "12px" }}>
          Streams audio to WhisperLiveKit, relays transcripts to backend for per-speaker language/profanity checks, and shows flags immediately.
        </p>
        <button className="button" style={{ marginTop: "10px" }} onClick={() => navigate("/teacher")}>
          Go to Teacher Dashboard
        </button>
      </div>

      <div className="card" style={{ minHeight: "360px" }}>
        <h3 style={{ marginTop: 0 }}>Live transcription (WLK)</h3>
        <TranscriptList transcripts={wlkTranscripts} />
        <h3 style={{ marginTop: "16px" }}>Backend stream</h3>
        <TranscriptList transcripts={analysisTranscripts} />
        <h3 style={{ marginTop: "16px" }}>Flags</h3>
        <div className="flag-panel">
          {flags.length === 0 && <div className="muted">No flags yet.</div>}
          {flags.map((f, idx) => (
            <div key={idx} className="flag-item">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <strong>{f.speaker || "unknown"}</strong>
                <span className="pill" style={{ background: "#fef9c3", color: "#92400e" }}>{f.lang}</span>
              </div>
              <div style={{ fontSize: "12px", marginTop: "4px" }} className="muted">
                {(f.alerts || []).map((a) => a.msg).join(" · ") || f.source}
              </div>
              <div style={{ marginTop: "6px" }}>{f.text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
