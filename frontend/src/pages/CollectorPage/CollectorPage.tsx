import { useState } from "react";
import "./CollectorPage.css";
import { CollectorSidebar } from "../../Components/CollectorSidebar/CollectorSidebar";
import { CollectorLocations } from "../../Components/CollectorLocations/CollectorLocations";
import { useNavigate } from "react-router-dom";
import { toastSucces } from "../../Model/toast";
import { logout } from "../../utility/api";
import { CollecterGetRequest } from '../../Components/ColectorReportRequest/ColectorReportRequest'
import ChatPage from "../ChatPage/ChatPage";
import ProfilePage from "../ProfilePage/ProfilePage";
import GarbageDetectionPage from "../GarbageDetectionPage/GarbageDetectionPage";

const CollectorPage = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("locations");

  const handleTabChange = (tab: string) => {
    if (tab === "Logout") {
      logout();
      navigate("/");
      toastSucces("Logout Successfully");
    } 
    setActiveTab(tab);
  };

  return (
    <div className="collector-page-wrapper">
      <CollectorSidebar activeTab={activeTab} onTabChange={handleTabChange} />
      <div className="collector-content">
        {activeTab === "locations" && <CollectorLocations />}
        {activeTab === "requestreport" && <CollecterGetRequest/>}
        {activeTab === "chat" && <ChatPage/>}
        {activeTab === "detect" && <GarbageDetectionPage/>}
        {activeTab === "profile" && <ProfilePage/>}
      </div>
    </div>
  );
};

export default CollectorPage;
