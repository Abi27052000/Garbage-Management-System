import { useState } from "react";
import "./AdminPage.css";
import { AdminSidebar } from "../../Components/AdminSidebar/AdminSidebar";
import { AdminReports } from "../../Components/AdminReports/AdminReports";
import { AdminCollectors } from "../../Components/AdminCollectors/AdminCollectors";
import {CollectorRegister}  from "../../Components/CollecterRegister/CollecterRegister";
import AdminTips from "../../Components/AdminTips/AdminTips";
import AdminQuizPage from "../../Components/AdminAddQuiz/AdminAddQuiz";
import { AdminLeaderboard } from "../../Components/AdminLeaderboard/AdminLeaderboard";
import GarbageDetectionPage from "../GarbageDetectionPage/GarbageDetectionPage";
import ChatPage from "../ChatPage/ChatPage";

export function AdminPage() {
  const [activeTab, setActiveTab] = useState("reports");

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

  return (
    <div className="admin-page-wrapper">
      <AdminSidebar activeTab={activeTab} onTabChange={handleTabChange} />
      <div className="admin-content">
        {activeTab === "reports" && <AdminReports />}
        {activeTab === "collectors" && <AdminCollectors />}
        {activeTab === "Register"  && <CollectorRegister/>}
        {activeTab === "TipsReview"  && <AdminTips/>}
        {activeTab === "AddQuiz"  && <AdminQuizPage/>}
        {activeTab === "Leaderboard"  && <AdminLeaderboard/>}
        {activeTab === "detect" && <GarbageDetectionPage />}
        {activeTab === "chat" && <ChatPage />}
      </div>
    </div>
  );
}

export default AdminPage;
