import { useEffect, useMemo, useRef, useState } from "react";
import GroupCard from "../components/GroupCard";
import TranscriptList from "../components/TranscriptList";
import { wsBase } from "../lib/config";

export default function TeacherPage() {
  const [groups, setGroups] = useState({});
  const [selectedGroup, setSelectedGroup] = useState("");
  const teacherSocketRef = useRef(null);
  const [allEvents, setAllEvents] = useState([]);

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
        dominance_state: data.dominance_state,
        speech_ratio: data.speech_ratio,
        silence: data.silence,
        target_topic: data.target_topic,
        target_description: data.target_description,
        source: data.source || "analysis",
        speaker: data.dominance_speaker || data.speaker,
        group_id: gid,
      };

      setAllEvents((prev) => [{ ...entry }, ...prev].slice(0, 200));

      setGroups((prev) => {
        const next = { ...prev };
        const existing = next[gid] || { transcripts: [], alertCount: 0 };
        const transcripts = [{ ...entry }, ...(existing.transcripts || [])].slice(0, 80);
        next[gid] = {
          last: entry,
          transcripts,
          alertCount: (existing.alertCount || 0) + (entry.alerts?.length || 1),
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
  const groupCount = teacherCards.length;

  return (
    <div className="grid grid-2" style={{ alignItems: "start" }}>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0 }}>Groups</h2>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <span className="pill" style={{ background: "#e0f2fe", color: "#075985" }}>
              Groups: {groupCount}
            </span>
            <span className="pill" style={{ background: "#e0f2fe", color: "#075985" }}>
              WS: {teacherSocketRef.current ? "connected" : "connecting..."}
            </span>
          </div>
        </div>
        <div className="grid grid-2" style={{ marginTop: "12px" }}>
          {teacherCards.length === 0 && <div className="muted">Waiting for groups...</div>}
          {teacherCards.map(([gid, data]) => (
            <GroupCard key={gid} groupId={gid} data={data} onSelect={setSelectedGroup} />
          ))}
        </div>
        <h3 style={{ marginTop: "12px" }}>All events</h3>
        <div className="list" style={{ maxHeight: "380px", overflowY: "auto" }}>
          {allEvents.length === 0 && <div className="muted">No events yet.</div>}
          {allEvents.map((ev, idx) => (
            <div
              key={idx}
              className="transcript"
              style={{ borderLeft: ev.alerts?.length ? "4px solid #f59e0b" : "4px solid #e2e8f0" }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <strong>{ev.group_id}</strong>
                <span className="pill" style={{ background: "#f1f5f9" }}>{ev.lang}</span>
                {ev.alerts?.length > 0 && (
                  <span className="pill" style={{ background: "#fef9c3", color: "#92400e" }}>Alert</span>
                )}
              </div>
              <div className="muted" style={{ fontSize: "12px" }}>{ev.speaker ? `Speaker ${ev.speaker}` : ""}</div>
              <div style={{ marginTop: "6px" }}>{ev.text}</div>
            </div>
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
