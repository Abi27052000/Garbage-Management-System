import React, { useEffect, useState } from "react";
import axios from "axios";
import "./CollectorLeaderboard.css";

interface CollectorLeaderboardUser {
  _id: string;
  collectorName: string;
  totalPoints: number;
  totalAssignments: number;
}

const CollectorLeaderboard: React.FC = () => {
  const [collectors, setCollectors] = useState<CollectorLeaderboardUser[]>([]);

  useEffect(() => {
    fetchLeaderboard();
  }, []);

  const fetchLeaderboard = async () => {
    try {
      const res = await axios.get("http://localhost:3000/api/collector/leaderboard");
      setCollectors(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  // Calculate percentage relative to highest points
  const maxPoints = collectors.length
    ? Math.max(...collectors.map((c) => c.totalPoints))
    : 1;

  return (
    <div className="leaderboard-container">
      <h2>🗑️ Collector Leaderboard</h2>

      {collectors.map((collector, index) => {
        const percentage = (collector.totalPoints / maxPoints) * 100;

        return (
          <div key={collector._id} className="leaderboard-item">
            <div className="leaderboard-header">
              <span className="rank">{index + 1}</span>
              <span className="name">{collector.collectorName}</span>
              <span className="score">{collector.totalPoints} <span>pts</span></span>
              <span className="percentage">{percentage.toFixed(0)}%</span>
            </div>

            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${percentage}%` }}
              ></div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default CollectorLeaderboard;