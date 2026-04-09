import React, { useEffect, useState } from "react";
import axios from "axios";
import "./ReporterLeaderboard.css";

interface LeaderboardUser {
  _id: string;
  username: string;
  totalCorrect: number;
  totalQuestions: number;
}

export const ReporterLeaderboard: React.FC = () => {
  const [users, setUsers] = useState<LeaderboardUser[]>([]);

  useEffect(() => {
    fetchLeaderboard();
  }, []);

  const fetchLeaderboard = async () => {
    try {
      const res = await axios.get("http://localhost:3000/api/quiz/leaderboard");
      setUsers(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  console.log(users);

  return (
    <div className="leaderboard-container">
      <h2>🏆 Leaderboard</h2>

      {users.map((user, index) => {
        const percentage =
          user.totalQuestions > 0
            ? (user.totalCorrect / user.totalQuestions) * 100
            : 0;

        return (
          <div key={user._id} className="leaderboard-item">
            <div className="leaderboard-header">
              <span className="rank">{index + 1}</span>

              <span className="name">{user.username}</span>

              <span className="score">
                {user.totalCorrect}/{user.totalQuestions}
              </span>

              <span className="percentage">
                {percentage.toFixed(0)}%
              </span>
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

 

export default ReporterLeaderboard;