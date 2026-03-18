import express from "express";
import Tip from "../models/tip.js";
const router = express.Router();

// Submit tip (USER)
router.post("/add", async (req, res) => {
    try {
        const { tip, user } = req.body;

        const newTip = new Tip({
            tip,
            user
        });

        await newTip.save();
        res.status(201).json({ message: "Tip submitted for approval" });

    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Get approved tips (USER VIEW)
router.get("/approved", async (req, res) => {
    try {
        const tips = await Tip.find({ status: "approved" }).sort({ createdAt: -1 });
        res.json(tips);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Get all tips (ADMIN)
router.get("/all", async (req, res) => {
    try {
        const tips = await Tip.find().sort({ createdAt: -1 });
        res.json(tips);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Approve tip
router.put("/approve/:id", async (req, res) => {
    try {
        await Tip.findByIdAndUpdate(req.params.id, { status: "approved" });
        res.json({ message: "Tip approved" });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});


router.put("/reject/:id", async (req, res) => {
    try {
        await Tip.findByIdAndUpdate(req.params.id, { status: "rejected" });
        res.json({ message: "Tip rejected" });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

export default router;