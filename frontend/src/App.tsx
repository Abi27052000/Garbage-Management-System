import { BrowserRouter, Routes, Route } from "react-router-dom";
import { DashBoard } from "./Components/DashBoard/DashBoard";
import { WasteageClassifier } from "./Components/WasteageClassifier/WasteageClassifier";
import { ReportGarbage } from "./Components/ReportGarbage/ReportGarbage";
import { HomePage } from "./pages/HomePage/HomePage";
import AdminPage from "./pages/AdminPage/AdminPage";
import CollectorPage from "./pages/CollectorPage/CollectorPage";
import ChatPage from "./pages/ChatPage/ChatPage";
import ProtectedRoute from "./Components/ProtectedRoute/ProtectedRoute";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { CollecterGetRequest } from "./Components/ColectorReportRequest/ColectorReportRequest";
import ProfilePage from "./pages/ProfilePage/ProfilePage";
import EducationPage from "./pages/EducationPage/EducationPage";
import { ReporterLeaderboard } from "./Components/ReporterLeaderboard/ReporterLeaderboard";



function App() {

  
  return (
    <>
      <ToastContainer />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/admin"
            element={
              <ProtectedRoute allowedRoles={["admin"]}>
                <AdminPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/collector"
            element={
              <ProtectedRoute allowedRoles={["collector"]}>
                <CollectorPage />
              </ProtectedRoute>
            }
          />

          <Route path="/reporter" element={
            <ProtectedRoute allowedRoles={["reporter"]}>
              <DashBoard />
            </ProtectedRoute>
          }
          >
            <Route path="wastage-classifier" element={<WasteageClassifier />} />
            <Route path="report-garbage" element={<ReportGarbage />} />
            <Route path="profile" element={<ProfilePage />} />
            <Route path="education" element={<EducationPage/>}/>
            <Route path="user-leader-board" element={<ReporterLeaderboard/>}/>
            <Route path="settings" element={<div>settings content here</div>} />
            <Route path="chat"
              element={
                <ProtectedRoute allowedRoles={["reporter", "collector", "admin"]}>
                  <ChatPage />
                </ProtectedRoute>
              } />
          </Route>
          <Route
            path="chat"
            element={
              <ProtectedRoute allowedRoles={["reporter", "collector", "admin"]}>
                <ChatPage />
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
