import React, { useEffect, useState } from "react";
import axios from "axios";
import "./AdminTips.css";

interface Tip {
    _id: string;
    tip: string;
    user: string;
    status: "pending" | "approved" | "rejected";
}

const AdminTips: React.FC = () => {
    const [tips, setTips] = useState<Tip[]>([]);
    const [filter, setFilter] = useState<"all" | "pending" | "approved" | "rejected">("pending");

    // 🔥 Fetch all tips
    useEffect(() => {
        fetchTips();
    }, []);

    const fetchTips = async () => {
        try {
            const res = await axios.get("http://localhost:3000/api/tips/all");
            setTips(res.data);
        } catch (error) {
            console.error("Error fetching tips:", error);
        }
    };

    // ✅ Approve Tip
    const approveTip = async (id: string) => {
        try {
            await axios.put(`http://localhost:3000/api/tips/approve/${id}`);
            fetchTips(); // refresh
        } catch (error) {
            console.error("Error approving tip:", error);
        }
    };

    // ❌ Reject Tip
    const rejectTip = async (id: string) => {
        try {
            await axios.put(`http://localhost:3000/api/tips/reject/${id}`);
            fetchTips(); // refresh
        } catch (error) {
            console.error("Error rejecting tip:", error);
        }
    };

    // 🎯 Filter logic
    const filteredTips =
        filter === "all"
            ? tips
            : tips.filter((tip) => tip.status === filter);

    return (
        <div className="admin-tips-container">
            <h2>Tip Management</h2>

            {/* 🔥 Filter Tabs */}
            <div className="tabs">
                <button onClick={() => setFilter("all")}>All</button>
                <button onClick={() => setFilter("pending")}>Pending</button>
                <button onClick={() => setFilter("approved")}>Approved</button>
                <button onClick={() => setFilter("rejected")}>Rejected</button>
            </div>

            {/* 📋 Tips List */}
            <div className="tips-list">
                {filteredTips.length === 0 ? (
                    <p>No tips found.</p>
                ) : (
                    filteredTips.map((tip) => (
                        <div key={tip._id} className="tip-card">
                            <p className="tip-text">"{tip.tip}"</p>
                            <p className="tip-user">- {tip.user}</p>
                            <p className={`status ${tip.status}`}>
                                {tip.status.toUpperCase()}
                            </p>

                            {/* 🔥 Action Buttons */}
                            {tip.status === "pending" && (
                                <div className="actions">
                                    <button
                                        className="approve-btn"
                                        onClick={() => approveTip(tip._id)}
                                    >
                                        Approve
                                    </button>

                                    <button
                                        className="reject-btn"
                                        onClick={() => rejectTip(tip._id)}
                                    >
                                        Reject
                                    </button>
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default AdminTips;