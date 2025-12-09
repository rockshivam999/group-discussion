import { useRef, useState } from "react";
import { franc } from "franc-min";
import TranscriptList from "../components/TranscriptList";
import { wsBase } from "../lib/config";
import { langName } from "../lib/utils";
import { useNavigate } from "react-router-dom";

export default function StudentPage() {
  const [groupId, setGroupId] = useState("Group-Alpha");
  const [topic, setTopic] = useState("Nature");
  const [description, setDescription] = useState("Discuss biodiversity, forests, oceans, and why protecting nature matters.");
  const [isRecording, setIsRecording] = useState(false);
  const [studentStatus, setStudentStatus] = useState("Idle");
  const [studentLangGuess, setStudentLangGuess] = useState("unknown");
  const mediaRecorderRef = useRef(null);
  const studentSocketRef = useRef(null);
  const [studentTranscripts, setStudentTranscripts] = useState([]);
  const navigate = useNavigate();

  const startRecording = async () => {
    const gid = groupId.trim();
    if (!gid || !topic.trim()) {
      alert("Please enter Group ID and Topic before starting.");
      return;
    }

    const socket = new WebSocket(`${wsBase}/ws/student/${encodeURIComponent(gid)}`);
    studentSocketRef.current = socket;
    setStudentStatus("Connecting...");

    socket.onopen = async () => {
      setStudentStatus("Streaming audio...");
      socket.send(
        JSON.stringify({
          topic: topic.trim(),
          description: description.trim(),
        })
      );
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredMime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType: preferredMime });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
          socket.send(event.data);
        }
      };
      recorder.onerror = (err) => {
        console.error("MediaRecorder error", err);
        setStudentStatus("Recorder error");
        stopRecording();
      };
      recorder.start(1000);
      setIsRecording(true);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "analysis") {
        const entry = {
          text: data.text,
          timestamp: data.timestamp,
          lang: data.lang,
          topic_score: data.topic_score,
          alerts: data.alerts || [],
          dominance: data.dominance_state,
          speech_ratio: data.speech_ratio,
          silence: data.silence,
          target_topic: data.target_topic,
        };
        setStudentTranscripts((prev) => [entry, ...prev].slice(0, 50));
        const guess = franc(data.text || "", { minLength: 3 });
        setStudentLangGuess(guess === "und" ? "unknown" : guess);
      }
    };

    socket.onclose = () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      }
      mediaRecorderRef.current = null;
      studentSocketRef.current = null;
      setIsRecording(false);
      setStudentStatus("Disconnected");
    };
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    }
    mediaRecorderRef.current = null;
    if (studentSocketRef.current) {
      studentSocketRef.current.close();
      studentSocketRef.current = null;
    }
    setIsRecording(false);
    setStudentStatus("Stopped");
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
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          {!isRecording ? (
            <button className="button primary" onClick={startRecording}>
              Start Recording
            </button>
          ) : (
            <button className="button danger" onClick={stopRecording}>
              Stop
            </button>
          )}
          <span className="pill" style={{ background: "#f1f5f9" }}>{studentStatus}</span>
        </div>
        <div style={{ marginTop: "12px" }}>
          <div className="pill" style={{ background: "#dcfce7", color: "#166534" }}>
            Client lang guess: {langName(studentLangGuess)}
          </div>
        </div>
        <p className="muted" style={{ marginTop: "12px" }}>
          Enter your Group ID and Topic before starting. Client guesses language locally; backend handles transcript,
          topic, dominance, profanity, and silence alerts.
        </p>
        <button className="button" style={{ marginTop: "10px" }} onClick={() => navigate("/teacher")}>
          Go to Teacher Dashboard
        </button>
      </div>
      <div className="card" style={{ minHeight: "360px" }}>
        <h3 style={{ marginTop: 0 }}>Transcript (live)</h3>
        <TranscriptList transcripts={studentTranscripts} />
      </div>
    </div>
  );
}
