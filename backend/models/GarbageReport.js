import mongoose from "mongoose";

const garbageReportSchema = new mongoose.Schema(
  {
    reporterName: {
      type: String,
      required: [true, "Reporter name is required"],
      trim: true,
    },

    reporter: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: false, // Optional if anonymous reporting is allowed
    },

    weight: {
      type: Number,
      required: [true, "Weight is required"],
      min: [0.1, "Weight must be greater than 0"],
    },

    points: {
      type: Number,
      default: function () {
        // 100 points per kg
        return Math.round(this.weight * 100);
      },
    },

    collectionDeadline: {
      type: Date,
      required: [true, "Collection deadline is required"],
      validate: {
        validator: function (value) {
          // Ensure deadline is at least 24 hours from now
          const tomorrow = new Date();
          tomorrow.setHours(tomorrow.getHours() + 24);
          return value >= tomorrow;
        },
        message: "Collection deadline must be at least 24 hours from now",
      },
    },

    additionalDetails: {
      type: String,
      trim: true,
      maxlength: [1000, "Additional details cannot exceed 1000 characters"],
    },

    garbageImage: {
      type: String, // Base64 encoded image or URL
    },

    location: {
      latitude: {
        type: Number,
        required: [true, "Latitude is required"],
      },
      longitude: {
        type: Number,
        required: [true, "Longitude is required"],
      },
      address: {
        type: String,
        required: [true, "Address is required"],
      },
    },

    status: {
      type: String,
      enum: ["pending", "in-progress", "collected", "rejected"],
      default: "pending",
    },

    collector: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: false,
    },

    assignedAt: {
      type: Date,
    },
  },
  {
    timestamps: true, 
  }
);

garbageReportSchema.index({ status: 1, createdAt: -1 });
garbageReportSchema.index({ collector: 1, status: 1 });
garbageReportSchema.index({ "location.latitude": 1, "location.longitude": 1 });


garbageReportSchema.pre("save", function (next) {
  if (this.isModified("weight")) {
    this.points = Math.round(this.weight * 100);
  }
  next();
});


garbageReportSchema.pre("save", function (next) {
  if (
    this.isModified("status") &&
    this.status === "collected" &&
    !this.collectedAt
  ) {
    this.collectedAt = new Date();
  }
  next();
});

// Middleware to set assignedAt when collector is assigned
garbageReportSchema.pre("save", function (next) {
  if (this.isModified("collector") && this.collector && !this.assignedAt) {
    this.assignedAt = new Date();
  }
  next();
});

const GarbageReport = mongoose.model("GarbageReport", garbageReportSchema);

export default GarbageReport;
