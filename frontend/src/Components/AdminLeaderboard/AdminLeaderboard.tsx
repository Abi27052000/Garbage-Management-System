import React, { useEffect, useState } from "react";
import axios from "axios";
import "./AdminLeaderboard.css";

interface UserLeaderboard {
  _id: string;
  username: string;
  totalCorrect: number;
  totalQuestions: number;
}

interface CollectorLeaderboard {
  _id: string;
  collectorName: string;
  totalPoints: number;
  totalAssignments: number;
}

export const AdminLeaderboard: React.FC = () => {
  const [userLeaderboard, setUserLeaderboard] = useState<UserLeaderboard[]>([]);
  const [collectorLeaderboard, setCollectorLeaderboard] = useState<CollectorLeaderboard[]>([]);

  useEffect(() => {
    fetchLeaderboards();
  }, []);

  const fetchLeaderboards = async () => {
    try {
      const [usersRes, collectorsRes] = await Promise.all([
        axios.get("http://localhost:3000/api/quiz/leaderboard"), // user quiz leaderboard
        axios.get("http://localhost:3000/api/collector/leaderboard"), // collector leaderboard
      ]);

      setUserLeaderboard(usersRes.data);
      setCollectorLeaderboard(collectorsRes.data);
    } catch (err) {
      console.error(err);
    }
  };

  const renderUserLeaderboard = () => {
    const maxScore = userLeaderboard.length
      ? Math.max(...userLeaderboard.map((u) => u.totalCorrect))
      : 1;

    return userLeaderboard.map((user, index) => {
      const percentage = (user.totalCorrect / user.totalQuestions) * 100;
      return (
        <div key={user._id} className="leaderboard-item">
          <div className="leaderboard-header">
            <span className="rank">{index + 1}</span>
            <span className="name">{user.username}</span>
            <span className="score">{user.totalCorrect}/{user.totalQuestions}</span>
            <span className="percentage">{percentage.toFixed(0)}%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${percentage}%` }}></div>
          </div>
        </div>
      );
    });
  };

  const renderCollectorLeaderboard = () => {
    const maxPoints = collectorLeaderboard.length
      ? Math.max(...collectorLeaderboard.map((c) => c.totalPoints))
      : 1;

    return collectorLeaderboard.map((collector, index) => {
      const percentage = (collector.totalPoints / maxPoints) * 100;
      return (
        <div key={collector._id} className="leaderboard-item">
          <div className="leaderboard-header">
            <span className="rank">{index + 1}</span>
            <span className="name">{collector.collectorName}</span>
            <span className="score">{collector.totalPoints} pts</span>
            <span className="percentage">{percentage.toFixed(0)}%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${percentage}%` }}></div>
          </div>
        </div>
      );
    });
  };

  return (
    <div className="admin-leaderboard-container">
      <h1>🏆 Admin Leaderboards</h1>

      <section>
        <h2>User Quiz Leaderboard</h2>
        {renderUserLeaderboard()}
      </section>

      <section>
        <h2>Collector Leaderboard</h2>
        {renderCollectorLeaderboard()}
      </section>
    </div>
  );
};

