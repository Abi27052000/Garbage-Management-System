import mongoose from "mongoose";

const tipSchema = new mongoose.Schema({
    tip: {
        type: String,
        required: true
    },
    user: {
        type: String,
        required: true
    },
    status: {
        type: String,
        enum: ["pending", "approved", "rejected"],
        default: "pending"
    }
}, { timestamps: true });

const Tips = mongoose.model("Tip", tipSchema);

export default Tips;