import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import { createServer } from "http";
import { Server } from "socket.io";
import authRouter from "./routers/authRouter.js";
import adminRouter from "./routers/adminRouter.js";
import garbageReportRouter from "./routers/garbageReportRouter.js";
import aiRecomandRouter from "./routers/aiRecomandRouter.js";
import chatRouter from "./routers/chatRouter.js";
import Chat from "./models/Chat.js";
import User from "./models/User.js";
import axios from "axios";
import tipRoutes from "./routers/tipRoutes.js";

dotenv.config();

const PORT = process.env.PORT;
const MONGODB_URI = process.env.MONGODB_URI;

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from uploads directory
app.use("/uploads", express.static("uploads"));



const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log("MongoDB Connected");
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
    process.exit(1);
  }
};

connectDB();

app.get("/", (req, res) => {
  res.json({
    message: "Garbage Management System API",
    status: "Running",
    port: PORT,
  });
});

app.use("/api/auth", authRouter);
app.use("/api/admin", adminRouter);
app.use("/api/garbage", garbageReportRouter);
app.use("/api/ask", aiRecomandRouter);
app.use("/api/chat", chatRouter);
app.use("/api/tips", tipRoutes);

import { GoogleGenerativeAI } from "@google/generative-ai";

app.get("/api/list-models", async (req, res) => {
  try {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const models = await genAI.listModels();
    res.json(models);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Start server
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*", // In production, specify your frontend URL
    methods: ["GET", "POST"],
  },
});

// Socket.IO connection handling
io.on("connection", (socket) => {
  console.log("User connected:", socket.id);

  // User joins with their user ID
  socket.on("join", async (userId) => {
    try {
      socket.userId = userId;
      socket.join(userId);

      // Update user online status
      await User.findByIdAndUpdate(userId, {
        isOnline: true,
        lastSeen: new Date(),
      });

      // Broadcast online status to all connected users
      socket.broadcast.emit("user_online", { userId });

      console.log(`User ${userId} is now online`);
    } catch (error) {
      console.error("Error on user join:", error);
    }
  });

  // Handle sending messages
  socket.on("send_message", async (data) => {
    try {
      const { senderId, receiverId, message } = data;

      // Save message to database
      const newMessage = new Chat({
        sender: senderId,
        receiver: receiverId,
        message,
      });
      await newMessage.save();

      // Populate sender info
      await newMessage.populate("sender", "username");

      // Send to receiver if online
      socket.to(receiverId).emit("receive_message", newMessage);

      // Send back to sender for confirmation
      socket.emit("message_sent", newMessage);
    } catch (error) {
      console.error("Error sending message:", error);
      socket.emit("message_error", { error: "Failed to send message" });
    }
  });

  // Handle typing indicators
  socket.on("typing", (data) => {
    const { receiverId, isTyping } = data;
    socket.to(receiverId).emit("user_typing", {
      userId: socket.userId,
      isTyping,
    });
  });

  // Handle marking messages as seen
  socket.on("mark_seen", async (data) => {
    try {
      const { senderId, receiverId } = data;

      await Chat.updateMany(
        { sender: senderId, receiver: receiverId, isSeen: false },
        { isSeen: true, seenAt: new Date() }
      );

      // Notify sender that messages were seen
      socket.to(senderId).emit("messages_seen", { receiverId });
    } catch (error) {
      console.error("Error marking messages as seen:", error);
    }
  });

  // Handle disconnection
  socket.on("disconnect", async () => {
    try {
      if (socket.userId) {
        // Update user offline status
        await User.findByIdAndUpdate(socket.userId, {
          isOnline: false,
          lastSeen: new Date(),
        });

        // Broadcast offline status
        socket.broadcast.emit("user_offline", {
          userId: socket.userId,
          lastSeen: new Date(),
        });

        console.log(`User ${socket.userId} disconnected`);
      }
    } catch (error) {
      console.error("Error on disconnect:", error);
    }
  });
});







app.post("/assign", async (req, res) => {
  const { lat, lon } = req.body;

  if (!lat || !lon) {
    return res.status(400).json({ error: "lat and lon are required" });
  }

  let nearestCollector = null;
  let minDistance = Infinity;

  for (const collector of collectors) {
    const toStart = await getDrivingDistance({ lat, lon }, collector.start);
    const toEnd = await getDrivingDistance({ lat, lon }, collector.end);

   


    const shortest = Math.min(toStart.distance, toEnd.distance);

    if (shortest < minDistance) {
      minDistance = shortest;
      nearestCollector = collector;
    }
  }

  if (nearestCollector) {
    res.json({
      message: `Nearest collector assigned: ${nearestCollector.name}`,
      collector: nearestCollector,
      distance_meters: minDistance,
    });
  } else {
    res.status(404).json({ error: "No collector found" });
  }
});





server.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
