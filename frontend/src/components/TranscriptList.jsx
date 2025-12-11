import { fmtTime } from "../lib/utils";

export default function TranscriptList({ transcripts, title }) {
  if (!transcripts || transcripts.length === 0) {
    return <div className="muted">No transcript yet.</div>;
  }

  return (
    <div className="list" style={{ marginTop: "8px", maxHeight: "420px", overflowY: "auto" }}>
      {title && <div style={{ fontWeight: 700 }}>{title}</div>}
      {transcripts.map((entry, idx) => (
        <div key={idx} className="transcript">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <strong>{fmtTime(entry.timestamp)}</strong>
            <span className="pill" style={{ background: "#f1f5f9" }}>{entry.lang}</span>
            {entry.source && (
              <span className="pill" style={{ background: "#ede9fe", color: "#5b21b6" }}>{entry.source}</span>
            )}
          </div>
          <div className="muted" style={{ fontSize: "12px" }}>
            {entry.target_topic ? `Topic: ${entry.target_topic}` : ""}
            {entry.speaker ? ` · Speaker: ${entry.speaker}` : ""}
          </div>
          <p style={{ margin: "6px 0" }}>{entry.text}</p>
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
            {(entry.alerts || []).map((a, aidx) => (
              <span
                key={aidx}
                className="pill"
                style={{
                  background: a.type === "OFFENSIVE" ? "#fee2e2" : "#fef9c3",
                  color: a.type === "OFFENSIVE" ? "#b91c1c" : "#92400e",
                }}
              >
                ⚠ {a.msg}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
