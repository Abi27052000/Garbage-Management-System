

import React, { useState, useEffect } from "react";
import axios from "axios";
import "./CommunityTips.css";
import { toastError, toastSucces, toastWarn } from "../../Model/toast";

interface Tip {
    _id: string;
    tip: string;
    user: string;
}

const CommunityTips: React.FC = () => {
    const [tips, setTips] = useState<Tip[]>([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [newTip, setNewTip] = useState("");
    const [newUser, setNewUser] = useState("");


    useEffect(() => {
        fetchTips();
    }, []);

    const fetchTips = async () => {
        try {
            const res = await axios.get("http://localhost:3000/api/tips/approved");
            setTips(res.data);
        } catch (error) {
            toastError("Error fetching tips");
            console.error("Error fetching tips:", error);
        }
    };

    const openModal = () => setIsModalOpen(true);

    const closeModal = () => {
        setIsModalOpen(false);
        setNewTip("");
        setNewUser("");
    };


    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (newTip.trim() === "" || newUser.trim() === "") return;

        try {
            await axios.post("http://localhost:3000/api/tips/add", {
                tip: newTip,
                user: newUser
            });

            toastSucces("Tip submitted! Waiting for admin approval");
            closeModal();

        } catch (error) {
            console.error("Error submitting tip:", error);
            toastError("Error submitting tip");
        }
    };

    return (
        <section className="community-tips-section">
            <h2>Share Your Tips! 💡</h2>
            <p>Inspire others! Submit your best recycling or composting tip.</p>

            <div className="submit-tip-button-container">
                <button onClick={openModal}>Submit Your Awareness Tip</button>
            </div>

            {/* Modal */}
            {isModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <h3>Submit Your Tip</h3>
                        <form onSubmit={handleSubmit}>
                            <textarea
                                placeholder="Enter your tip..."
                                value={newTip}
                                onChange={(e) => setNewTip(e.target.value)}
                                required
                            />
                            <input
                                type="text"
                                placeholder="Your Name"
                                value={newUser}
                                onChange={(e) => setNewUser(e.target.value)}
                                required
                            />
                            <div className="modal-buttons">
                                <button type="submit">Submit</button>
                                <button type="button" onClick={closeModal}>
                                    Cancel
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            <h3>Community Tips:</h3>

            <div className="tips-list">
                {tips.length === 0 ? (
                    <p>No approved tips yet.</p>
                ) : (
                    tips.map((tip) => (
                        <div key={tip._id} className="tip-card">
                            <p className="tip-text">"{tip.tip}"</p>
                            <p className="tip-user">- {tip.user}</p>
                        </div>
                    ))
                )}
            </div>
        </section>
    );
};

export default CommunityTips;