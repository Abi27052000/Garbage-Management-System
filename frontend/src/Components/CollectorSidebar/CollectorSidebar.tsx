import "./CollectorSidebar.css";
import {
  FaMapMarkerAlt,
  FaUser,
  FaSignOutAlt,
  FaComments,
  FaSearch,
} from "react-icons/fa";
import { FaCodePullRequest } from "react-icons/fa6";
import awarenessImage from "../../assets/images/awareness_of_waste_recycling.png";

interface CollectorSidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export function CollectorSidebar({
  activeTab,
  onTabChange,
}: CollectorSidebarProps) {
  const menuItems = [
    {
      id: "locations",
      label: "Locations",
      icon: <FaMapMarkerAlt />,
    },
    {
      id: "chat",
      label: "Chat",
      icon: <FaComments />,
    },
     {
      id: "detect",
      label: "Detect Garbage",
      icon: <FaSearch />,
    },
    {
      id: "requestreport",
      label: "Request report",
      icon: <FaCodePullRequest />,
    },
    {
      id: "profile",
      label: "Profile",
      icon: <FaUser/>,
    },
    {
      id: "Logout",
      label: "Logout",
      icon: <FaSignOutAlt />,
    },
  ];

  return (
    <div className="collector-sidebar">
      <div className="collector-sidebar-header">
        <div className="collector-header-content">
          <FaUser className="collector-icon" />
          <h2>Collector Panel</h2>
        </div>
      </div>

      <div className="collector-sidebar-img">
        <img src={awarenessImage} alt="Collector Dashboard" />
      </div>

      <nav className="collector-nav">
        <ul>
          {menuItems.map((item) => (
            <li key={item.id}>
              <button
                className={activeTab === item.id ? "active-link" : ""}
                onClick={() => onTabChange(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
