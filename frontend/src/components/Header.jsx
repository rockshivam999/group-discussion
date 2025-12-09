import { Link, useLocation } from "react-router-dom";

export default function Header() {
  const location = useLocation();
  const isTeacher = location.pathname.startsWith("/teacher");
  const isStudent = location.pathname.startsWith("/student");

  return (
    <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <div>
        <h1 style={{ margin: 0, fontSize: "28px" }}>Classroom Sentinel</h1>
        <p className="muted" style={{ margin: 0 }}>
          Real-time group discussion monitor (FastAPI + React)
        </p>
      </div>
      <div className="tabs">
        <Link className={`tab ${isTeacher ? "active" : ""}`} to="/teacher">
          Teacher
        </Link>
        <Link className={`tab ${isStudent ? "active" : ""}`} to="/student">
          Student
        </Link>
      </div>
    </header>
  );
}
