import { useState } from "react";
import "./AdminPage.css";
import { AdminSidebar } from "../../Components/AdminSidebar/AdminSidebar";
import { AdminReports } from "../../Components/AdminReports/AdminReports";
import { AdminCollectors } from "../../Components/AdminCollectors/AdminCollectors";
import {CollectorRegister}  from "../../Components/CollecterRegister/CollecterRegister";
import { useNavigate } from "react-router-dom";
import AdminTips from "../../Components/AdminTips/AdminTips";

export function AdminPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("reports");

  const handleTabChange = (tab: string) => {
    if (tab === "chat") {
      navigate("/chat");
    } else {
      setActiveTab(tab);
    }
  };

  return (
    <div className="admin-page-wrapper">
      <AdminSidebar activeTab={activeTab} onTabChange={handleTabChange} />
      <div className="admin-content">
        {activeTab === "reports" && <AdminReports />}
        {activeTab === "collectors" && <AdminCollectors />}
        {activeTab === "Register"  && <CollectorRegister/>}
        {activeTab === "TipsReview"  && <AdminTips/>}
      </div>
    </div>
  );
}

export default AdminPage;
