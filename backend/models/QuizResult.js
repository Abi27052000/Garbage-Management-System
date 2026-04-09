// // models/QuizResult.js
// import mongoose from "mongoose";

// const quizResultSchema = new mongoose.Schema({
//   userId: { type: String, required: true },
//   username: { type: String, required: true },
//   quizId: { type: mongoose.Schema.Types.ObjectId, ref: "Quiz", required: true },
//   answers: [
//     {
//       questionId: { type: mongoose.Schema.Types.ObjectId, ref: "Quiz" },
//       selectedOption: String,
//       correct: Boolean
//     }
//   ],
//   score: { type: Number, default: 0 },
// }, { timestamps: true });

// const QuizResult = mongoose.model("QuizResult", quizResultSchema);
// export default QuizResult;

import mongoose from "mongoose";

const quizResultSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  username: { type: String, required: true },

  // ❗ removed required (important fix)
  quizId: { type: mongoose.Schema.Types.ObjectId, ref: "Quiz" },

  answers: [
    {
      questionId: { type: mongoose.Schema.Types.ObjectId, ref: "Quiz" },
      selectedOption: String,
      correct: Boolean
    }
  ],

  score: { type: Number, default: 0 },
}, { timestamps: true });

const QuizResult = mongoose.model("QuizResult", quizResultSchema);
export default QuizResult;