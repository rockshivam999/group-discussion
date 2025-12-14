import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const defaultWsUrl = import.meta.env?.VITE_WS_URL || "ws://localhost:8100/asr";
const backendWsUrl =
  import.meta.env?.VITE_BACKEND_WS ||
  (() => {
    if (typeof window === "undefined") return "ws://localhost:8000/monitor";
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.hostname || "localhost";
    return `${protocol}://${host}:8000/monitor`;
  })();

const connectionLabel = {
  connected: "Connected",
  connecting: "Connecting…",
  disconnected: "Disconnected",
  error: "Error"
};

const allowedLanguageOptions = [
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "hi", label: "Hindi" },
  { value: "zh", label: "Chinese" },
  { value: "auto", label: "Auto-detect" }
];

function App() {
  const [wsUrl, setWsUrl] = useState(defaultWsUrl);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [connectionError, setConnectionError] = useState("");
  const [serverMode, setServerMode] = useState("");
  const [serverState, setServerState] = useState("Idle");
  const [conversation, setConversation] = useState([]);
  const [lastPhrase, setLastPhrase] = useState("");
  const [chunkWindow, setChunkWindow] = useState([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [batchAggregate, setBatchAggregate] = useState("");
  const [liveBuffer, setLiveBuffer] = useState({ diarization: "", transcription: "", translation: "" });
  const [isRecording, setIsRecording] = useState(false);
  const [backendStatus, setBackendStatus] = useState("disconnected");
  const [flaggedItems, setFlaggedItems] = useState([]);
  const [topic, setTopic] = useState("Nature");
  const [contextText, setContextText] = useState("Mountains, Sea, Air, Clouds");
  const [allowedLanguage, setAllowedLanguage] = useState("en");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [teacherHistory, setTeacherHistory] = useState([]);
  const socketRef = useRef(null);
  const backendSocketRef = useRef(null);
  const backendReconnectRef = useRef(null);
  const fullTextRef = useRef("");
  const lastPhraseRef = useRef("");
  const chunkWindowRef = useRef([]);
  const historyRef = useRef("");
  const conversationRef = useRef([]);
  const mediaStreamRef = useRef(null);
  const recorderRef = useRef(null);

  const backendHttpBase = useMemo(() => {
    try {
      const url = new URL(backendWsUrl);
      url.protocol = url.protocol === "wss:" ? "https:" : "http:";
      url.pathname = "/";
      return url.toString().replace(/\/$/, "");
    } catch (err) {
      return "";
    }
  }, []);

  const conversationHistoryText = useMemo(() => {
    return conversation
      .map((line) => {
        const speaker = line.speaker !== undefined ? line.speaker : "?";
        const main = line.text || "";
        const translation = line.translation ? ` [${line.translation}]` : "";
        return `S${speaker}: ${main}${translation}`;
      })
      .join("\n");
  }, [conversation]);

  const teacherConversationText = useMemo(() => {
    const lines = teacherHistory.length ? teacherHistory : conversation;
    return lines
      .map((line) => {
        const speaker = line.speaker !== undefined ? line.speaker : "?";
        const main = line.text || "";
        const translation = line.translation ? ` [${line.translation}]` : "";
        return `S${speaker}: ${main}${translation}`;
      })
      .join("\n");
  }, [conversation, teacherHistory]);

  const sendDeltaToBackend = useCallback((payload) => {
    const socket = backendSocketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    try {
      socket.send(JSON.stringify(payload));
    } catch (err) {
      console.warn("Could not send delta to backend", err);
    }
  }, []);

  const determineAggregateLanguage = useCallback((lines = []) => {
    const counts = lines.reduce((acc, line) => {
      const lang = (line.detected_language || "").toLowerCase().trim();
      if (!lang) return acc;
      acc[lang] = (acc[lang] || 0) + 1;
      return acc;
    }, {});
    let topLang = "";
    let topCount = 0;
    Object.entries(counts).forEach(([lang, count]) => {
      if (count > topCount) {
        topLang = lang;
        topCount = count;
      }
    });
    return topLang || "";
  }, []);

  const mockAnalyzeSnapshot = useCallback((snapshot, meta) => {
    const speakers = Array.from(new Set(snapshot.map((l) => l.speaker).filter((s) => s !== undefined)));
    const totalLines = snapshot.length;
    const participationBalance =
      speakers.length > 1
        ? `Voices observed: ${speakers.join(", ")}. Distribution looks balanced over ${totalLines} turns.`
        : `Single dominant voice across ${totalLines} turns.`;
    const topicNote = meta.topic
      ? `Appears mostly on topic "${meta.topic}" with light drift detected.`
      : "Topic not set; cannot evaluate adherence.";
    return {
      participation_balance: participationBalance,
      topic_adherence: topicNote,
      summary: "Mock analysis placeholder. Replace with external LLM call."
    };
  }, []);

  const updateLatestChunk = useCallback((historyText, lastLineMeta = null) => {
    const previous = fullTextRef.current || "";
    const trimmed = historyText.trim();

    // Only consider the newly appended portion to avoid re-counting the full transcript.
    let delta = "";
    if (trimmed && trimmed.startsWith(previous)) {
      delta = trimmed.slice(previous.length).trim();
    } else {
      delta = trimmed;
    }

    fullTextRef.current = trimmed;
    historyRef.current = trimmed;
    
    if (!delta) return;

    console.log("Latest transcript addition:", delta);
    if (lastLineMeta) {
      sendDeltaToBackend({
        type: "delta",
        text: delta,
        speaker: lastLineMeta.speaker ?? null,
        start: lastLineMeta.start,
        end: lastLineMeta.end,
        detected_language: lastLineMeta.detected_language,
        topic,
        context: contextText,
        allowed_language: allowedLanguage,
        timestamp: lastLineMeta.timestamp || new Date().toISOString()
      });
    }

    setChunkTotal((count) => count + 1);
    setChunkWindow((prev) => {
      const next = [...prev, delta];
      chunkWindowRef.current = next;

      if (next.length >= 10) {
        const aggregate = next.join(" ");
        setBatchAggregate(aggregate);
        console.log("10-chunk aggregate ready (then reset):", aggregate);
        chunkWindowRef.current = [];
        return [];
      }

      return next;
    });
  }, [allowedLanguage, contextText, sendDeltaToBackend, topic]);

  const sendSnapshotToBackend = useCallback(() => {
    const socket = backendSocketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    const snapshot = (conversationRef.current || []).map((line) => ({
      speaker: line.speaker,
      text: line.text,
      start: line.start,
      end: line.end,
      detected_language: line.detected_language,
      translation: line.translation,
      timestamp: line.timestamp || new Date().toISOString(),
      allowed_language: allowedLanguage,
      topic,
      context: contextText
    }));

    const aggregateDetectedLanguage = determineAggregateLanguage(snapshot);
    const payload = {
      type: "snapshot",
      items: snapshot,
      topic,
      context: contextText,
      allowed_language: allowedLanguage,
      detected_language: aggregateDetectedLanguage
    };

    try {
      socket.send(JSON.stringify(payload));
    } catch (err) {
      console.warn("Could not send snapshot to backend", err);
    }
    setTeacherHistory(snapshot);
    setAnalysisResult(mockAnalyzeSnapshot(snapshot, payload));
  }, [allowedLanguage, contextText, determineAggregateLanguage, mockAnalyzeSnapshot, topic]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current) {
      try {
        recorderRef.current.stop();
      } catch (err) {
        console.warn("Error stopping recorder", err);
      }
      recorderRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setIsRecording(false);
  }, []);

  const startRecording = useCallback(async () => {
    if (isRecording) return;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Microphone access is not available in this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      let recorder;
      try {
        recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      } catch (err) {
        recorder = new MediaRecorder(stream);
      }
      recorderRef.current = recorder;

      recorder.ondataavailable = (evt) => {
        if (!evt.data || evt.data.size === 0) return;
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
          socketRef.current.send(evt.data);
        }
      };

      recorder.onstop = () => {
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        recorderRef.current = null;
        setIsRecording(false);
      };

      recorder.start(250);
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone access error", err);
      setConnectionError("Microphone access denied or unavailable.");
      stopRecording();
    }
  }, [isRecording, stopRecording]);

  const handleMessage = useCallback(
    (event) => {
      try {
        const payload = JSON.parse(event.data);

        if (payload.type === "config") {
          setServerMode(payload.useAudioWorklet ? "AudioWorklet (PCM)" : "MediaRecorder (WebM)");
          return;
        }

        if (payload.type === "ready_to_stop") {
          setServerState("Ready to stop");
          return;
        }

        const {
          lines = [],
          buffer_transcription = "",
          buffer_diarization = "",
          buffer_translation = "",
          status = "active_transcription"
        } = payload;

        setServerState(status);
        setLiveBuffer({
          diarization: buffer_diarization || "",
          transcription: buffer_transcription || "",
          translation: buffer_translation || ""
        });

        const merged = lines.map((line, idx) => {
          const isLatest = idx === lines.length - 1;
          const textParts = [line.text || ""];
          if (isLatest) {
            if (buffer_diarization) textParts.push(buffer_diarization);
            if (buffer_transcription) textParts.push(buffer_transcription);
          }

          const translationParts = [];
          if (line.translation) translationParts.push(line.translation);
          if (isLatest && buffer_translation) translationParts.push(buffer_translation);

          return {
            ...line,
            is_loading: line.speaker === 0 && status !== "no_audio_detected",
            text: textParts.join(" ").replace(/\s+/g, " ").trim(),
            translation: translationParts.join(" ").replace(/\s+/g, " ").trim()
          };
        });

        const effectiveLines = merged.filter((l) => {
          const main = (l.text || "").trim();
          const translation = (l.translation || "").trim();
          return main.length > 0 || translation.length > 0;
        });

        if (effectiveLines.length === 0) {
          return;
        }

        setConversation(effectiveLines);

        const historyText = effectiveLines
          .map((l) => {
            const speaker = l.speaker !== undefined ? l.speaker : "?";
            const translation = l.translation ? ` [${l.translation}]` : "";
            return `S${speaker}: ${l.text}${translation}`;
          })
          .join(" | ");

        const latestStableMeta = (() => {
          for (let i = effectiveLines.length - 1; i >= 0; i--) {
            const line = effectiveLines[i];
            const t = (line.text || "").trim();
            if (!t) continue;
            if (line.speaker === 0 || line.speaker === -2) continue; // skip loading/silence
            return line;
          }
          return null;
        })();

        if (latestStableMeta) {
          const latestStableText = (latestStableMeta.text || "").trim();
          if (latestStableText && latestStableText !== lastPhraseRef.current) {
            lastPhraseRef.current = latestStableText;
            setLastPhrase(latestStableText);
          }
        }

        updateLatestChunk(historyText, effectiveLines[effectiveLines.length - 1]);
      } catch (err) {
        console.error("Could not parse websocket payload", err);
      }
    },
    [updateLatestChunk]
  );

  const connect = useCallback(() => {
    if (connectionStatus === "connected" || connectionStatus === "connecting") return;
    if (!wsUrl.startsWith("ws://") && !wsUrl.startsWith("wss://")) {
      setConnectionError("WebSocket URL must start with ws:// or wss://");
      setConnectionStatus("error");
      return;
    }

    setConnectionStatus("connecting");
    setConnectionError("");
    fullTextRef.current = "";
    historyRef.current = "";
    setLastPhrase("");
    lastPhraseRef.current = "";
    setChunkWindow([]);
    setChunkTotal(0);
    setBatchAggregate("");
    chunkWindowRef.current = [];

    try {
      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        setConnectionStatus("connected");
        setServerState("Listening");
        startRecording();
      };

      socket.onerror = (err) => {
        setConnectionStatus("error");
        setConnectionError("WebSocket connection failed. Check the URL and server status.");
        console.error("WebSocket error", err);
      };

      socket.onclose = (event) => {
        setConnectionStatus("disconnected");
        console.warn("WebSocket closed", { code: event.code, reason: event.reason, wasClean: event.wasClean });
        socketRef.current = null;
        stopRecording();
      };

      socket.onmessage = handleMessage;
    } catch (err) {
      setConnectionStatus("error");
      setConnectionError("Could not open WebSocket connection.");
    }
  }, [connectionStatus, handleMessage, startRecording, stopRecording, wsUrl]);

  const connectBackend = useCallback(() => {
    if (backendSocketRef.current && backendSocketRef.current.readyState === WebSocket.OPEN) return;

    try {
      const socket = new WebSocket(backendWsUrl);
      backendSocketRef.current = socket;
      setBackendStatus("connecting");

      socket.onopen = () => {
        setBackendStatus("connected");
        sendSnapshotToBackend();
      };

      socket.onerror = (err) => {
        console.warn("Backend websocket error", err);
        setBackendStatus("error");
      };

      socket.onclose = () => {
        setBackendStatus("disconnected");
        backendSocketRef.current = null;
        if (backendReconnectRef.current) clearTimeout(backendReconnectRef.current);
        backendReconnectRef.current = setTimeout(connectBackend, 3000);
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === "flagged" && payload.payload) {
            setFlaggedItems((prev) => [...prev, payload.payload]);
          }
          if (payload.type === "flagged_bulk" && Array.isArray(payload.items)) {
            setFlaggedItems(payload.items);
          }
          if (payload.type === "history" && payload.history) {
            setTeacherHistory(payload.history);
          }
        } catch (err) {
          console.warn("Could not parse backend message", err);
        }
      };
    } catch (err) {
      console.warn("Could not open backend websocket", err);
      setBackendStatus("error");
    }
  }, [backendWsUrl, sendSnapshotToBackend]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    stopRecording();
    setConnectionStatus("disconnected");
    setServerState("Idle");
    setConversation([]);
    setLiveBuffer({ diarization: "", transcription: "", translation: "" });
    setLastPhrase("");
    lastPhraseRef.current = "";
    setBatchAggregate("");
    setChunkWindow([]);
    setChunkTotal(0);
    chunkWindowRef.current = [];
    fullTextRef.current = "";
    historyRef.current = "";
  }, [stopRecording]);

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      stopRecording();
      if (backendSocketRef.current) {
        backendSocketRef.current.close();
      }
      if (backendReconnectRef.current) {
        clearTimeout(backendReconnectRef.current);
      }
    };
  }, [stopRecording]);

  useEffect(() => {
    conversationRef.current = conversation;
  }, [conversation]);

  useEffect(() => {
    connectBackend();
    return () => {
      if (backendSocketRef.current) backendSocketRef.current.close();
      if (backendReconnectRef.current) clearTimeout(backendReconnectRef.current);
    };
  }, [connectBackend]);

  useEffect(() => {
    if (!backendHttpBase || backendStatus !== "connected") return;
    const fetchData = async () => {
      try {
        const flaggedRes = await fetch(`${backendHttpBase}/flagged`);
        const flaggedJson = await flaggedRes.json();
        if (Array.isArray(flaggedJson.flagged)) {
          setFlaggedItems(flaggedJson.flagged);
        }
        if (flaggedJson.meta) {
          if (flaggedJson.meta.topic) setTopic(flaggedJson.meta.topic);
          if (flaggedJson.meta.context) setContextText(flaggedJson.meta.context);
          if (flaggedJson.meta.allowed_language) setAllowedLanguage(flaggedJson.meta.allowed_language);
        }
      } catch (err) {
        console.warn("Could not fetch flagged items", err);
      }

      try {
        const historyRes = await fetch(`${backendHttpBase}/complete-conversation`);
        const historyJson = await historyRes.json();
        if (Array.isArray(historyJson.history)) {
          setTeacherHistory(historyJson.history);
        }
      } catch (err) {
        console.warn("Could not fetch conversation history", err);
      }
    };
    fetchData();
  }, [backendHttpBase, backendStatus]);

  useEffect(() => {
    const interval = setInterval(() => {
      sendSnapshotToBackend();
    }, 30000);
    return () => clearInterval(interval);
  }, [sendSnapshotToBackend]);

  const resetCounters = () => {
    setLastPhrase("");
    lastPhraseRef.current = "";
    setChunkWindow([]);
    setChunkTotal(0);
    setBatchAggregate("");
    chunkWindowRef.current = [];
    fullTextRef.current = historyRef.current;
  };

  return (
    <div className="app-shell">
      <header className="page-header">
        <h1 className="page-title">Classroom Monitor</h1>
        <p className="muted">
          Listening to WhisperLiveKit over WebSocket and tracking speaker-aware transcripts, the latest word/phrase, and
          10-chunk aggregations.
        </p>
      </header>

      <section className="panel connection-bar">
        <input
          className="input-field"
          value={wsUrl}
          onChange={(e) => setWsUrl(e.target.value)}
          placeholder="ws://host:port/asr"
        />
        <div className="connection-actions">
          <span className="status-chip">
            <span className={`status-dot ${connectionStatus}`} />
            {connectionLabel[connectionStatus]}
          </span>
          <button
            className="btn btn-primary"
            onClick={connectionStatus === "connected" ? disconnect : connect}
            disabled={connectionStatus === "connecting"}
          >
            {connectionStatus === "connected" ? "Disconnect" : "Connect"}
          </button>
          <button
            className="btn btn-primary"
            onClick={isRecording ? stopRecording : startRecording}
            disabled={connectionStatus !== "connected"}
          >
            {isRecording ? "Stop mic" : "Start mic"}
          </button>
          <button className="btn btn-ghost" onClick={resetCounters}>
            Reset counters
          </button>
        </div>
      </section>

      <section className="panel meta-bar">
        <div className="field">
          <label className="field-label">Discussion topic</label>
          <input
            className="input-field"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g., Nature"
          />
        </div>
        <div className="field">
          <label className="field-label">Context</label>
          <input
            className="input-field"
            value={contextText}
            onChange={(e) => setContextText(e.target.value)}
            placeholder="Mountains, Sea, Air, Clouds…"
          />
        </div>
        <div className="field">
          <label className="field-label">Allowed language</label>
          <select
            className="input-field"
            value={allowedLanguage}
            onChange={(e) => setAllowedLanguage(e.target.value)}
          >
            {allowedLanguageOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </section>

      {connectionError && <div className="live-note">{connectionError}</div>}

      <div className="grid">
        <section className="panel transcript">
          <div className="badge-row">
          <span className="status-chip">
            <span className="status-dot connected" />
            {serverState}
          </span>
          {serverMode && <span className="pill muted">{serverMode}</span>}
          <span className="pill muted">Backend WS: {backendStatus}</span>
          {lastPhrase && <span className="pill">Last phrase: {lastPhrase}</span>}
        </div>

          <div className="scrollable">
            {conversation.length === 0 && <p className="muted">Waiting for live transcription…</p>}
            {conversation.map((line, idx) => {
              const speakerLabel = line.speaker === -2 ? "…" : line.speaker ?? "?";
              return (
                <article key={`${line.start || idx}-${idx}`} className="line">
                  <div className="speaker-badge">{speakerLabel}</div>
                  <div className="line-content">
                    <p className="line-text">{line.text || "…"}</p>
                    {line.translation && <p className="translation">Translation: {line.translation}</p>}
                    <div className="badge-row">
                      {line.start !== undefined && line.end !== undefined && (
                        <span className="pill muted">
                          {line.start} – {line.end}
                        </span>
                      )}
                      {line.detected_language && <span className="pill">Lang: {line.detected_language}</span>}
                      {line.is_loading && <span className="pill muted">Processing…</span>}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="panel">
          <h3 className="card-title">Stream Stats</h3>
          <div className="stat">
            <span className="stat-label">Last word / phrase</span>
            <span className="stat-value">{lastPhrase || "—"}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Chunks in current batch</span>
            <span className="stat-value">
              {chunkWindow.length} / 10
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Total chunks observed</span>
            <span className="stat-value">{chunkTotal}</span>
          </div>
          {batchAggregate ? (
            <div className="live-note">Last 10-chunk aggregate (then reset): {batchAggregate}</div>
          ) : (
            <div className="live-note">
              Aggregating every 10 new chunks so you can run rolling analysis without a long diff.
            </div>
          )}

          <div className="history">
            <p className="history-title">Conversation history (chronological)</p>
            <p className="history-text">{conversationHistoryText || "No transcript yet."}</p>
          </div>

          <div className="live-note">
            <strong>Live buffers</strong>
            <br />
            Diarization: {liveBuffer.diarization || "—"}
            <br />
            Transcription: {liveBuffer.transcription || "—"}
            <br />
            Translation: {liveBuffer.translation || "—"}
          </div>

          <div className="live-note">
            <strong>Flagged language ({flaggedItems.length})</strong>
            {flaggedItems.length === 0 ? (
              <div>No issues flagged yet.</div>
            ) : (
              <ul style={{ margin: "6px 0 0", paddingLeft: "18px" }}>
                {flaggedItems.slice(-5).map((item, idx) => (
                  <li key={`${item.timestamp || idx}-${idx}`}>
                    Speaker {item.speaker ?? "?"}: {item.text}{" "}
                    {item.flagged_words ? `(bad: ${item.flagged_words.join(", ")})` : ""}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>

      <section className="panel teacher-dashboard">
        <div className="teacher-header">
          <div>
            <h3 className="card-title">Teacher Dashboard</h3>
            <p className="muted small">Moderation, full conversation, and automated insights</p>
          </div>
          <span className="pill muted">Auto-refresh every 30s</span>
        </div>
        <div className="teacher-grid">
          <div className="teacher-card">
            <h4 className="card-title">Flagged items ({flaggedItems.length})</h4>
            {flaggedItems.length === 0 ? (
              <p className="muted">No flagged language or mismatches yet.</p>
            ) : (
              <ul className="flagged-list">
                {flaggedItems.map((item, idx) => (
                  <li key={`${item.timestamp || idx}-${idx}`}>
                    <div className="flagged-line">
                      <span className="pill">Speaker {item.speaker ?? "?"}</span>
                      {item.flagged_reason === "language_mismatch" ? (
                        <span className="pill muted">
                          Language mismatch ({item.detected_language || "?"} vs allowed {item.allowed_language || "?"})
                        </span>
                      ) : (
                        <span className="pill muted">Profanity</span>
                      )}
                    </div>
                    <div className="flagged-text">{item.text}</div>
                    {item.flagged_words && (
                      <div className="muted small">Words: {item.flagged_words.join(", ")}</div>
                    )}
                    <div className="muted small">
                      Topic: {item.topic || "n/a"} | Context: {item.context || "n/a"}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="teacher-card">
            <h4 className="card-title">Complete conversation</h4>
            <div className="history">
              <p className="history-title">Timeline</p>
              <p className="history-text">{teacherConversationText || "No transcript yet."}</p>
            </div>
          </div>
          <div className="teacher-card">
            <h4 className="card-title">30s Analysis (mock)</h4>
            {analysisResult ? (
              <div className="analysis">
                <p className="analysis-line"><strong>Participation balance:</strong> {analysisResult.participation_balance}</p>
                <p className="analysis-line"><strong>Topic adherence:</strong> {analysisResult.topic_adherence}</p>
                <p className="muted small">{analysisResult.summary}</p>
              </div>
            ) : (
              <p className="muted">Waiting for first 30s snapshot…</p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
