import { useState, useEffect } from "react";
import "./AdminReports.css";
import {
  FaTrash,
  FaMapMarkerAlt,
  FaWeight,
  FaClock,
  FaUser,
  FaCheckCircle,
  FaHourglassHalf,
  FaTimes,
  FaFilter,
} from "react-icons/fa";

interface Location {
  latitude: number;
  longitude: number;
  address: string;
}

interface GarbageReport {
  _id: string;
  reporterName: string;
  weight: number;
  points: number;
  collectionDeadline: string;
  additionalDetails?: string;
  garbageImage?: string;
  location: Location;
  status: "pending" | "in-progress" | "collected" | "rejected";
  collector?: {
    _id: string;
    name: string;
    email: string;
  };
  collectedAt?: string;
  rejectionReason?: string;
  assignedAt?: string;
  createdAt: string;
  updatedAt: string;
}

type FilterType = "all" | "pending" | "in-progress" | "collected" | "rejected";

export function AdminReports() {
  const [reports, setReports] = useState<GarbageReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<FilterType>("all");
  const [selectedReport, setSelectedReport] = useState<GarbageReport | null>(
    null
  );
  const [showModal, setShowModal] = useState(false);

  // Fetch reports from backend
  const fetchReports = async (status?: string) => {
  setLoading(true);
  setError(null);
  try {
    const url =
      status && status !== "all"
        ? `http://localhost:3000/api/garbage/reports?status=${status}`
        : "http://localhost:3000/api/garbage/reports";

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error("Failed to fetch reports");
    }

    const data = await response.json();
    setReports(Array.isArray(data) ? data : (data.data || [])); // handle both array and paginated response
  } catch (err) {
    setError(err instanceof Error ? err.message : "An error occurred");
    console.error("Error fetching reports:", err);
  } finally {
    setLoading(false);
  }
};

  useEffect(() => {
    fetchReports(activeFilter === "all" ? undefined : activeFilter);
  }, [activeFilter]);

  const handleFilterChange = (filter: FilterType) => {
    setActiveFilter(filter);
  };

  const handleViewDetails = (report: GarbageReport) => {
    setSelectedReport(report);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedReport(null);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pending":
        return <FaClock className="status-icon pending-icon" />;
      case "in-progress":
        return <FaHourglassHalf className="status-icon in-progress-icon" />;
      case "collected":
        return <FaCheckCircle className="status-icon collected-icon" />;
      case "rejected":
        return <FaTimes className="status-icon rejected-icon" />;
      default:
        return null;
    }
  };

  const getStatusClass = (status: string) => {
    return `status-badge status-${status}`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="admin-reports-container">
      <div className="admin-reports-header">
        <div className="admin-reports-header-content">
          <FaTrash className="admin-reports-header-icon" />
          <h2>Garbage Reports Management</h2>
        </div>
        <p className="admin-reports-subtitle">
          Monitor and manage all garbage collection reports
        </p>
      </div>

      {/* Filter Buttons */}
      <div className="filter-section">
        <div className="filter-label">
          <FaFilter className="filter-icon" />
          <span>Filter by Status:</span>
        </div>
        <div className="filter-buttons">
          <button
            className={`filter-btn ${activeFilter === "all" ? "active" : ""}`}
            onClick={() => handleFilterChange("all")}
          >
            All Reports
            <span className="filter-count">{reports.length}</span>
          </button>
          <button
            className={`filter-btn ${
              activeFilter === "pending" ? "active" : ""
            }`}
            onClick={() => handleFilterChange("pending")}
          >
            <FaClock className="btn-icon" />
            Pending
            <span className="filter-count">
              {reports.filter((r) => r.status === "pending").length}
            </span>
          </button>
          <button
            className={`filter-btn ${
              activeFilter === "in-progress" ? "active" : ""
            }`}
            onClick={() => handleFilterChange("in-progress")}
          >
            <FaHourglassHalf className="btn-icon" />
            In Progress
            <span className="filter-count">
              {reports.filter((r) => r.status === "in-progress").length}
            </span>
          </button>
          <button
            className={`filter-btn ${
              activeFilter === "collected" ? "active" : ""
            }`}
            onClick={() => handleFilterChange("collected")}
          >
            <FaCheckCircle className="btn-icon" />
            Collected
            <span className="filter-count">
              {reports.filter((r) => r.status === "collected").length}
            </span>
          </button>
          <button
            className={`filter-btn ${
              activeFilter === "rejected" ? "active" : ""
            }`}
            onClick={() => handleFilterChange("rejected")}
          >
            <FaTimes className="btn-icon" />
            Rejected
            <span className="filter-count">
              {reports.filter((r) => r.status === "rejected").length}
            </span>
          </button>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading reports...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="error-container">
          <FaTimes className="error-icon" />
          <p>Error: {error}</p>
          <button
            onClick={() =>
              fetchReports(activeFilter === "all" ? undefined : activeFilter)
            }
            className="retry-btn"
          >
            Retry
          </button>
        </div>
      )}

      {/* Reports Grid */}
      {!loading && !error && reports.length > 0 && (
        <div className="reports-grid">
          {reports.map((report) => (
            <div key={report._id} className="report-card">
              <div className="report-card-header">
                <div className="report-id">
                  <span className="id-label">Report #</span>
                  <span className="id-value">{report._id.slice(-6)}</span>
                </div>
                <div className={getStatusClass(report.status)}>
                  {getStatusIcon(report.status)}
                  {report.status}
                </div>
              </div>

              {report.garbageImage && (
                <div className="report-image">
                  <img
                    src={`http://localhost:3000/${report.garbageImage}`}
                    alt="Garbage"
                  />
                </div>
              )}

              <div className="report-card-body">
                <div className="report-info-row">
                  <FaUser className="info-icon" />
                  <span className="info-label">Reporter:</span>
                  <span className="info-value">{report.reporterName}</span>
                </div>

                <div className="report-info-row">
                  <FaWeight className="info-icon" />
                  <span className="info-label">Weight:</span>
                  <span className="info-value">{report.weight} kg</span>
                </div>

                <div className="report-info-row">
                  <FaMapMarkerAlt className="info-icon" />
                  <span className="info-label">Location:</span>
                  <span className="info-value info-address">
                    {report.location.address}
                  </span>
                </div>

                <div className="report-info-row">
                  <FaClock className="info-icon" />
                  <span className="info-label">Deadline:</span>
                  <span className="info-value">
                    {formatDate(report.collectionDeadline)}
                  </span>
                </div>

                {report.collector && (
                  <div className="report-info-row">
                    <FaUser className="info-icon" />
                    <span className="info-label">Collector:</span>
                    <span className="info-value">{report.collector.name}</span>
                  </div>
                )}
              </div>

              <div className="report-card-footer">
                <button
                  className="view-details-btn"
                  onClick={() => handleViewDetails(report)}
                >
                  View Details
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && reports.length === 0 && (
        <div className="empty-state">
          <FaTrash className="empty-icon" />
          <h3>No Reports Found</h3>
          <p>There are no garbage reports matching the selected filter.</p>
        </div>
      )}

      {/* Modal for Report Details */}
      {showModal && selectedReport && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Report Details</h3>
              <button className="modal-close-btn" onClick={closeModal}>
                <FaTimes />
              </button>
            </div>

            <div className="modal-body">
              <div className="modal-section">
                <h4>Report Information</h4>
                <div className="modal-info-grid">
                  <div className="modal-info-item">
                    <span className="modal-label">Report ID:</span>
                    <span className="modal-value">{selectedReport._id}</span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Status:</span>
                    <span className={getStatusClass(selectedReport.status)}>
                      {getStatusIcon(selectedReport.status)}
                      {selectedReport.status}
                    </span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Reporter Name:</span>
                    <span className="modal-value">
                      {selectedReport.reporterName}
                    </span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Weight:</span>
                    <span className="modal-value">
                      {selectedReport.weight} kg
                    </span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Points:</span>
                    <span className="modal-value">{selectedReport.points}</span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Collection Deadline:</span>
                    <span className="modal-value">
                      {formatDate(selectedReport.collectionDeadline)}
                    </span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Created At:</span>
                    <span className="modal-value">
                      {formatDate(selectedReport.createdAt)}
                    </span>
                  </div>
                </div>
              </div>

              {selectedReport.garbageImage && (
                <div className="modal-section">
                  <h4>Garbage Image</h4>
                  <div className="modal-image">
                    <img
                      src={`http://localhost:3000/${selectedReport.garbageImage}`}
                      alt="Garbage"
                    />
                  </div>
                </div>
              )}

              <div className="modal-section">
                <h4>Location Details</h4>
                <div className="modal-info-grid">
                  <div className="modal-info-item">
                    <span className="modal-label">Address:</span>
                    <span className="modal-value">
                      {selectedReport.location.address}
                    </span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Latitude:</span>
                    <span className="modal-value">
                      {selectedReport.location.latitude}
                    </span>
                  </div>
                  <div className="modal-info-item">
                    <span className="modal-label">Longitude:</span>
                    <span className="modal-value">
                      {selectedReport.location.longitude}
                    </span>
                  </div>
                </div>
              </div>

              {selectedReport.additionalDetails && (
                <div className="modal-section">
                  <h4>Additional Details</h4>
                  <p className="modal-description">
                    {selectedReport.additionalDetails}
                  </p>
                </div>
              )}

              {selectedReport.collector && (
                <div className="modal-section">
                  <h4>Collector Information</h4>
                  <div className="modal-info-grid">
                    <div className="modal-info-item">
                      <span className="modal-label">Name:</span>
                      <span className="modal-value">
                        {selectedReport.collector.name}
                      </span>
                    </div>
                    <div className="modal-info-item">
                      <span className="modal-label">Email:</span>
                      <span className="modal-value">
                        {selectedReport.collector.email}
                      </span>
                    </div>
                    {selectedReport.assignedAt && (
                      <div className="modal-info-item">
                        <span className="modal-label">Assigned At:</span>
                        <span className="modal-value">
                          {formatDate(selectedReport.assignedAt)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {selectedReport.collectedAt && (
                <div className="modal-section">
                  <h4>Collection Information</h4>
                  <div className="modal-info-item">
                    <span className="modal-label">Collected At:</span>
                    <span className="modal-value">
                      {formatDate(selectedReport.collectedAt)}
                    </span>
                  </div>
                </div>
              )}

              {selectedReport.rejectionReason && (
                <div className="modal-section">
                  <h4>Rejection Reason</h4>
                  <p className="modal-description rejection-reason">
                    {selectedReport.rejectionReason}
                  </p>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="modal-btn close-modal-btn"
                onClick={closeModal}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AdminReports;
