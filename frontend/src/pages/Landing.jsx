import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="grid grid-2" style={{ alignItems: "center" }}>
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Welcome to Classroom Sentinel</h2>
        <p className="muted">
          Pick a view: Teacher dashboard to monitor all groups, or Student to start a group mic with topic context for
          off-topic checks.
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
        <h3 style={{ marginTop: 0 }}>Tips</h3>
        <ul className="muted" style={{ paddingLeft: "18px" }}>
          <li>Student view asks for Group ID, Topic, and a short description.</li>
          <li>Teacher dashboard shows live transcripts, topic score, dominance, and alerts.</li>
          <li>Open both routes in separate tabs/windows to monitor live.</li>
        </ul>
      </div>
    </div>
  );
}
