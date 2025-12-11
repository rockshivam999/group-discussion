import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const defaultWsUrl = import.meta.env?.VITE_WS_URL || "ws://localhost:8100/asr";

const connectionLabel = {
  connected: "Connected",
  connecting: "Connecting…",
  disconnected: "Disconnected",
  error: "Error"
};

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
  const socketRef = useRef(null);
  const fullTextRef = useRef("");
  const lastPhraseRef = useRef("");
  const chunkWindowRef = useRef([]);
  const historyRef = useRef("");
  const mediaStreamRef = useRef(null);
  const recorderRef = useRef(null);

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

  const updateLatestChunk = useCallback((historyText) => {
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
  }, []);

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

        const latestStableText = (() => {
          for (let i = effectiveLines.length - 1; i >= 0; i--) {
            const t = (effectiveLines[i].text || "").trim();
            const speaker = effectiveLines[i].speaker;
            if (!t) continue;
            if (speaker === 0 || speaker === -2) continue; // skip loading/silence
            return t;
          }
          return "";
        })();

        if (latestStableText && latestStableText !== lastPhraseRef.current) {
          lastPhraseRef.current = latestStableText;
          setLastPhrase(latestStableText);
        }

        updateLatestChunk(historyText);
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
    };
  }, [stopRecording]);

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

      {connectionError && <div className="live-note">{connectionError}</div>}

      <div className="grid">
        <section className="panel transcript">
          <div className="badge-row">
            <span className="status-chip">
              <span className="status-dot connected" />
              {serverState}
            </span>
            {serverMode && <span className="pill muted">{serverMode}</span>}
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
        </section>
      </div>
    </div>
  );
}

export default App;
