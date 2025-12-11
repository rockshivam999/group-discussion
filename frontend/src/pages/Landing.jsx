import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="grid grid-2" style={{ alignItems: "center" }}>
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Welcome to Classroom Sentinel</h2>
        <p className="muted">
          Two live dashboards: Teacher monitors all groups; Student opens mic, sees live transcription, and flags when
          language or profanity issues occur.
        </p>
        <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginTop: "12px" }}>
          <Link className="button primary" to="/teacher">
            Open Teacher Dashboard
          </Link>
          <Link className="button" style={{ background: "#0f172a", color: "#fff" }} to="/student">
            Open Student Mic
          </Link>
        </div>
      </div>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>What happens</h3>
        <ul className="muted" style={{ paddingLeft: "18px" }}>
          <li>Student streams mic to WhisperLiveKit and forwards transcripts to backend.</li>
          <li>Backend tracks per-speaker words; every +10 words triggers language check; every chunk checks profanity.</li>
          <li>LLM stub runs every 60s to emit dominance/off-topic insights.</li>
        </ul>
      </div>
    </div>
  );
}
