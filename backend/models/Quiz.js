
import mongoose from "mongoose";

const optionSchema = new mongoose.Schema({
  text: { type: String, required: true },
  isCorrect: { type: Boolean, default: false }
});

const quizSchema = new mongoose.Schema({
  question: { type: String, required: true },
  options: [optionSchema],
  createdBy: { type: String }, 
}, { timestamps: true });

const Quiz = mongoose.model("Quiz", quizSchema);
export default Quiz;