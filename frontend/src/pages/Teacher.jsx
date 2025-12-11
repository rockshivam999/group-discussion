import { useEffect, useMemo, useRef, useState } from "react";
import GroupCard from "../components/GroupCard";
import TranscriptList from "../components/TranscriptList";
import { wsBase } from "../lib/config";

export default function TeacherPage() {
  const [groups, setGroups] = useState({});
  const [selectedGroup, setSelectedGroup] = useState("");
  const teacherSocketRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket(`${wsBase}/ws/teacher`);
    teacherSocketRef.current = socket;

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (!data.group_id) return;
      const gid = data.group_id;

      const entry = {
        text: data.text,
        lang: data.lang,
        timestamp: data.timestamp,
        topic_score: data.topic_score,
        alerts: data.alerts || [],
        dominance: data.dominance_state,
        speech_ratio: data.speech_ratio,
        silence: data.silence,
        target_topic: data.target_topic,
        target_description: data.target_description,
        source: data.source || "analysis",
        speaker: data.dominance_speaker,
      };

      const hasFlag =
        (entry.alerts && entry.alerts.length > 0) ||
        entry.dominance === "DOMINATING" ||
        entry.dominance === "QUIET";

      if (!hasFlag) return;

      setGroups((prev) => {
        const next = { ...prev };
        const existing = next[gid] || { transcripts: [] };
        const transcripts = [{ ...entry }, ...(existing.transcripts || [])].slice(0, 40);
        next[gid] = {
          last: entry,
          transcripts,
        };
        return next;
      });
      if (!selectedGroup) setSelectedGroup(gid);
    };

    socket.onclose = () => {
      teacherSocketRef.current = null;
    };

    return () => socket.close();
  }, [selectedGroup]);

  const teacherCards = useMemo(() => Object.entries(groups), [groups]);
  const transcripts = selectedGroup ? groups[selectedGroup]?.transcripts : [];

  return (
    <div className="grid grid-2" style={{ alignItems: "start" }}>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0 }}>Groups</h2>
          <span className="pill" style={{ background: "#e0f2fe", color: "#075985" }}>
            Live WS: {teacherSocketRef.current ? "connected" : "connecting..."}
          </span>
        </div>
        <div className="grid grid-2" style={{ marginTop: "12px" }}>
          {teacherCards.length === 0 && <div className="muted">Waiting for groups...</div>}
          {teacherCards.map(([gid, data]) => (
            <GroupCard key={gid} groupId={gid} data={data} onSelect={setSelectedGroup} />
          ))}
        </div>
      </div>

      <div className="card" style={{ minHeight: "380px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0 }}>Transcript & Alerts</h2>
          <select
            value={selectedGroup}
            onChange={(e) => setSelectedGroup(e.target.value)}
            style={{ padding: "8px", borderRadius: "8px", border: "1px solid #e2e8f0" }}
          >
            <option value="">Select group</option>
            {teacherCards.map(([gid]) => (
              <option key={gid} value={gid}>
                {gid}
              </option>
            ))}
          </select>
        </div>
        {!selectedGroup && <div className="muted" style={{ marginTop: "12px" }}>Pick a group to inspect details.</div>}
        {selectedGroup && <TranscriptList transcripts={transcripts} />}
      </div>
    </div>
  );
}
