import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Header from "./components/Header";
import Landing from "./pages/Landing";
import TeacherPage from "./pages/Teacher";
import StudentPage from "./pages/Student";

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <Header />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/teacher" element={<TeacherPage />} />
          <Route path="/student" element={<StudentPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
