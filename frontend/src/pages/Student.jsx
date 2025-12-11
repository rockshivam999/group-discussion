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
  const lastSentRef = useRef({});
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
        const entry = {
          text: data.text,
          timestamp: data.timestamp,
          lang: data.lang,
          topic_score: data.topic_score,
          alerts,
          dominance: data.dominance_state,
          speech_ratio: data.speech_ratio,
          silence: data.silence,
          target_topic: data.target_topic,
          source: data.source || "analysis",
          speaker: data.dominance_speaker || data.speaker,
        };
        setAnalysisTranscripts((prev) => [entry, ...prev].slice(0, 200));
        if ((data.alerts || []).length > 0 || data.source === "llm") {
          setFlags((prev) => [entry, ...prev].slice(0, 120));
        }
      };

      const wlkSocket = new WebSocket(session.wlk_ws_url);
      wlkSocketRef.current = wlkSocket;
      setStudentStatus("Connecting to WhisperLiveKit...");

      wlkSocket.onopen = async () => {
        setStudentStatus("Streaming to WhisperLiveKit...");
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

        const lines = (payload.lines || []).filter((l) => l.text && l.text.trim());
        if (lines.length === 0) return;

        const mapped = lines.map((l) => ({
          text: cleanText(l.text || ""),
          speaker: l.speaker ?? "unknown",
          lang: l.detected_language || payload.lang || payload.language || "unknown",
          timestamp: Date.now() / 1000,
          source: "wlk",
        }));
        setWlkTranscripts((prev) => [...mapped.reverse(), ...prev].slice(0, 200));

        const buffers = speakerBuffersRef.current;
        mapped.forEach((entry) => {
          const words = entry.text.split(/\s+/).filter(Boolean);
          const wordCount = words.length;
          const buf = buffers[entry.speaker] || { wordCount: 0, ts: 0, text: "" };
          const newWords = Math.max(0, wordCount - buf.wordCount);
          const timeGap = entry.timestamp - (buf.ts || 0);
          const lastSent = lastSentRef.current[entry.speaker] || { len: 0, ts: 0 };
          const charGrowth = entry.text.length - (lastSent.len || 0);

          const shouldSend =
            !buf.text ||
            newWords >= WORD_THRESHOLD ||
            timeGap > SILENCE_SPLIT_SECONDS ||
            charGrowth >= 20;

          if (shouldSend) {
            ingestToBackend(entry, payload);
            buffers[entry.speaker] = { wordCount, ts: entry.timestamp, text: entry.text };
            lastSentRef.current[entry.speaker] = { len: entry.text.length, ts: entry.timestamp };
          }
        });
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
