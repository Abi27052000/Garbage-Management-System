

import GarbageReport from "../models/GarbageReport.js";
import CollectorAssignment from "../models/CollectorAssignment.js";
import { findNearestCollector } from "../Utility/getDrivingDistance.js";


export const createGarbageReport = async (req, res) => {
  try {
    const {
      reporterName,
      weight,
      collectionDeadline,
      additionalDetails,
      latitude,
      longitude,
      address,
    } = req.body;

    const garbageImage = req.file ? req.file.path : null;

    if (!reporterName || !weight || !collectionDeadline || !latitude || !longitude || !address) {
      return res.status(400).json({ message: "Missing required fields" });
    }

    const location = { latitude, longitude, address };

    const report = await GarbageReport.create({
      reporterName,
      reporter: req.user?._id || null,
      weight,
      collectionDeadline,
      additionalDetails,
      garbageImage,
      location,
      status: "in-progress",
    });


    const nearest = await findNearestCollector({ lat: latitude, lon: longitude });
    console.log("NEAREST =", nearest);


    if (nearest && nearest.collector) {
      const assignment = await CollectorAssignment.create({
        collector: nearest.collector._id,
        report: report._id,
        distance: nearest.distance,
        duration: nearest.duration,
        nearestPoint: nearest.nearestPoint
      });


      report.collector = nearest.collector._id;
      await report.save();

      return res.status(201).json({
        success: true,
        message: `Report submitted and assigned to ${nearest.collector.name}`,
        data: { report, assignment },
      });
    }

    res.status(201).json({
      success: true,
      message: "Report submitted but no collector found nearby",
      data: report,
    });
  } catch (error) {
    console.error("Error creating report:", error);
    res.status(500).json({ success: false, message: error.message });
  }
};



export const getAllGarbageReports = async (req, res) => {
  try {
    const {
      status,
      collector,
      page = 1,
      limit = 10,
      sortBy = "-createdAt",
    } = req.query;

    const query = {};

    if (status) {
      query.status = status;
    }

    if (collector) {
      query.collector = collector;
    }

    const skip = (page - 1) * limit;

    const reports = await GarbageReport.find(query)
      .populate("collector", "username email")
      .populate("reporter", "username email")
      .sort(sortBy)
      .skip(skip)
      .limit(parseInt(limit));

    const total = await GarbageReport.countDocuments(query);

    res.status(200).json({
      success: true,
      data: reports,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
  console.error("Error fetching garbage reports:", error);
  res.status(500).json({
    success: false,
    message: error.message, // <-- show actual error
  });
}
  
};

// Get single garbage report by ID
export const getGarbageReportById = async (req, res) => {
  try {
    const { id } = req.params;

    const report = await GarbageReport.findById(id)
      .populate("collector", "username email")
      .populate("reporter", "username email");

    if (!report) {
      return res.status(404).json({
        success: false,
        message: "Garbage report not found",
      });
    }

    res.status(200).json({
      success: true,
      data: report,
    });
  } catch (error) {
    console.error("Error fetching garbage report:", error);
    res.status(500).json({
      success: false,
      message: "Failed to fetch garbage report",
    });
  }
};

// Update garbage report status
export const updateGarbageReportStatus = async (req, res) => {
  try {
    const { id } = req.params;
    const { status, rejectionReason } = req.body;

    const validStatuses = ["pending", "in-progress", "collected", "rejected"];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({
        success: false,
        message: "Invalid status value",
      });
    }

    const report = await GarbageReport.findById(id);

    if (!report) {
      return res.status(404).json({
        success: false,
        message: "Garbage report not found",
      });
    }

    report.status = status;

    if (status === "rejected" && rejectionReason) {
      report.rejectionReason = rejectionReason;
    }

    await report.save();

    res.status(200).json({
      success: true,
      message: "Garbage report status updated successfully",
      data: report,
    });
  } catch (error) {
    console.error("Error updating garbage report status:", error);
    res.status(500).json({
      success: false,
      message: "Failed to update garbage report status",
    });
  }
};

// Assign collector to garbage report
export const assignCollector = async (req, res) => {
  try {
    const { id } = req.params;
    const { collectorId } = req.body;

    if (!collectorId) {
      return res.status(400).json({
        success: false,
        message: "Collector ID is required",
      });
    }

    const report = await GarbageReport.findById(id);

    if (!report) {
      return res.status(404).json({
        success: false,
        message: "Garbage report not found",
      });
    }

    report.collector = collectorId;
    report.status = "in-progress";

    await report.save();

    res.status(200).json({
      success: true,
      message: "Collector assigned successfully",
      data: report,
    });
  } catch (error) {
    console.error("Error assigning collector:", error);
    res.status(500).json({
      success: false,
      message: "Failed to assign collector",
    });
  }
};

// Delete garbage report
export const deleteGarbageReport = async (req, res) => {
  try {
    const { id } = req.params;

    const report = await GarbageReport.findByIdAndDelete(id);

    if (!report) {
      return res.status(404).json({
        success: false,
        message: "Garbage report not found",
      });
    }

    res.status(200).json({
      success: true,
      message: "Garbage report deleted successfully",
    });
  } catch (error) {
    console.error("Error deleting garbage report:", error);
    res.status(500).json({
      success: false,
      message: "Failed to delete garbage report",
    });
  }
};

// Get reports by collector
export const getReportsByCollector = async (req, res) => {
  try {
    const { collectorId } = req.params;
    const { status, page = 1, limit = 10 } = req.query;

    const query = { collector: collectorId };

    if (status) {
      query.status = status;
    }

    const skip = (page - 1) * limit;

    const reports = await GarbageReport.find(query)
      .sort("-createdAt")
      .skip(skip)
      .limit(parseInt(limit));

    const total = await GarbageReport.countDocuments(query);

    res.status(200).json({
      success: true,
      data: reports,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error("Error fetching collector reports:", error);
    res.status(500).json({
      success: false,
      message: "Failed to fetch collector reports",
    });
  }
};

// Get statistics
export const getReportStatistics = async (req, res) => {
  try {
    const totalReports = await GarbageReport.countDocuments();
    const pendingReports = await GarbageReport.countDocuments({
      status: "pending",
    });
    const inProgressReports = await GarbageReport.countDocuments({
      status: "in-progress",
    });
    const collectedReports = await GarbageReport.countDocuments({
      status: "collected",
    });
    const rejectedReports = await GarbageReport.countDocuments({
      status: "rejected",
    });

    const totalWeight = await GarbageReport.aggregate([
      { $match: { status: "collected" } },
      { $group: { _id: null, total: { $sum: "$weight" } } },
    ]);

    const totalPoints = await GarbageReport.aggregate([
      { $match: { status: "collected" } },
      { $group: { _id: null, total: { $sum: "$points" } } },
    ]);

    res.status(200).json({
      success: true,
      data: {
        totalReports,
        pendingReports,
        inProgressReports,
        collectedReports,
        rejectedReports,
        totalWeightCollected: totalWeight[0]?.total || 0,
        totalPointsAwarded: totalPoints[0]?.total || 0,
      },
    });
  } catch (error) {
    console.error("Error fetching statistics:", error);
    res.status(500).json({
      success: false,
      message: "Failed to fetch statistics",
    });
  }
};

export const getAssignGarbageForCollector = async (req, res) => {
  try {
    const { collecter_id } = req.params;
    if (!collecter_id) {
      res.status(201).json({
        success: false,
        message: "Coller id is must",
      })
    }

    const assignments = await CollectorAssignment.find({ collector: collecter_id });

    if (!assignments || assignments.length === 0) {
      return res.status(200).json({
        success: false,
        message: "No reports assigned for this collector",
        report_data: [],
      });
    }

    const combinedData = await Promise.all(
      assignments.map(async (report) => {
        const reportDetails = await GarbageReport.findById(report.report)
        return {
          assignmentId: report._id,
          assignedAt: report.assignedAt,
          distance: report.distance,
          duration: report.duration,
          status: report.status,
          nearestPoint: report.nearestPoint,
          report: reportDetails ? reportDetails : null,
        };
      })
    )

    res.status(200).json({
      success: true,
      report_data: combinedData
    })



  } catch (err) {
    console.error("Error fetching statistics:", err);
    res.status(500).json({
      success: false,
      message: "Failed to fetch reports for collecter",
    });
  }
}


