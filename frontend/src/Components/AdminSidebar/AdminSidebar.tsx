import "./AdminSidebar.css";
import { FaFileAlt, FaUsers, FaUserShield, FaComments, FaSearch } from "react-icons/fa";
import awarenessImage from "../../assets/images/awareness_of_waste_recycling.png";
import { GiArchiveRegister } from "react-icons/gi";
import { MdOutlineTipsAndUpdates } from "react-icons/md";

interface AdminSidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export function AdminSidebar({ activeTab, onTabChange }: AdminSidebarProps) {
  const menuItems = [
    {
      id: "reports",
      label: "Reports",
      icon: <FaFileAlt />,
    },
    {
      id: "TipsReview",
      label: "Tips Review",
      icon: <MdOutlineTipsAndUpdates />,
    },
    {
      id: "collectors",
      label: "Collectors",
      icon: <FaUsers />,
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
      id: "Register",
      label: "Register",
      icon: < GiArchiveRegister />,
    },
  ];

  return (
    <div className="admin-sidebar">
      <div className="admin-sidebar-header">
        <div className="admin-header-content">
          <FaUserShield className="admin-icon" />
          <h2>Admin Panel</h2>
        </div>
      </div>

      <div className="admin-sidebar-img">
        <img src={awarenessImage} alt="Admin Dashboard" />
      </div>

      <nav className="admin-nav">
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
