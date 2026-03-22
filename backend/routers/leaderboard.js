// routes/collectorRoutes.js
import express from "express";
import CollectorAssignment from "../models/CollectorAssignment.js";
import GarbageReport from "../models/GarbageReport.js";
import Collector from "../models/Collector.js";

const router = express.Router();

// GET collector leaderboard based on cumulative points
router.get("/leaderboard", async (req, res) => {
  try {
    // Fetch all collector assignments and populate collector and report
    const assignments = await CollectorAssignment.find()
      .populate("collector")
      .populate("report"); // populate the linked garbage report to get points

    // Map to calculate cumulative points
    const collectorPointsMap = {};

    assignments.forEach((assignment) => {
      if (!assignment.collector || !assignment.report) return;

      const collectorId = assignment.collector._id.toString();
      const points = assignment.report.points || 0;

      if (!collectorPointsMap[collectorId]) {
        collectorPointsMap[collectorId] = {
          collectorName: assignment.collector.name,
          totalPoints: 0,
          totalAssignments: 0,
        };
      }

      collectorPointsMap[collectorId].totalPoints += points;
      collectorPointsMap[collectorId].totalAssignments += 1;
    });

    // Convert map to array and sort by totalPoints descending
    const leaderboard = Object.keys(collectorPointsMap).map((id) => ({
      _id: id,
      ...collectorPointsMap[id],
    }));

    leaderboard.sort((a, b) => b.totalPoints - a.totalPoints);

    res.json(leaderboard);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

export default router;