import { fmtTime } from "../lib/utils";

export default function GroupCard({ groupId, data, onSelect }) {
  const borderColor = data.last?.alerts?.some((a) => a.type === "OFFENSIVE")
    ? "#ef4444"
    : data.last?.alerts?.some((a) => a.type === "LANGUAGE")
    ? "#f59e0b"
    : data.last?.alerts?.some((a) => a.type?.includes("TOPIC"))
    ? "#fb923c"
    : "#22c55e";

  return (
    <div className="card" style={{ borderColor, borderWidth: 2, borderStyle: "solid" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong>{groupId}</strong>
        <span className="pill" style={{ background: "#f1f5f9" }}>{data.last?.lang || "??"}</span>
        {data.last?.source && (
          <span className="pill" style={{ background: "#ede9fe", color: "#5b21b6" }}>{data.last.source}</span>
        )}
      </div>
      <div className="muted" style={{ fontSize: "12px" }}>
        Topic: {data.last?.target_topic || "—"}
      </div>
      <div className="muted" style={{ fontSize: "12px" }}>
        {data.last ? fmtTime(data.last.timestamp) : "—"}
      </div>
      <p style={{ margin: "8px 0 6px 0" }}>{data.last?.text || <span className="muted">Waiting…</span>}</p>
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        <span className="pill" style={{ background: "#e0f2fe", color: "#075985" }}>
          Topic {data.last?.topic_score || ""}
        </span>
        {data.last?.dominance_state && (
          <span className="pill" style={{ background: "#dcfce7", color: "#166534" }}>
            {data.last?.dominance_state} · {data.last?.speech_ratio ?? 0}
          </span>
        )}
      </div>
      <div className="list" style={{ marginTop: "6px" }}>
        {(data.last?.alerts || []).map((a, idx) => (
          <span
            key={idx}
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
      <button className="button primary" style={{ marginTop: "10px" }} onClick={() => onSelect(groupId)}>
        View transcript
      </button>
    </div>
  );
}
