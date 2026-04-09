// routes/quizRoutes.js
import express from "express";
import Quiz from "../models/Quiz.js";
import QuizResult from "../models/QuizResult.js";

const router = express.Router();

// Admin adds a quiz
router.post("/admin/add", async (req, res) => {
  try {
    const { question, options, createdBy } = req.body;
    const quiz = new Quiz({ question, options, createdBy });
    await quiz.save();
    res.status(201).json({ message: "Quiz added successfully", quiz });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get all quizzes (for users)
router.get("/all", async (req, res) => {
  try {
    const quizzes = await Quiz.find();
    res.json(quizzes);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post("/submit", async (req, res) => {
  try {
    const { userId, username, answers } = req.body;

    if (!answers || answers.length === 0) {
      return res.status(400).json({ error: "No answers submitted" });
    }

    let totalScore = 0;
    const answerDetails = [];

    for (const userAns of answers) {
      const quiz = await Quiz.findById(userAns.quizId);
      if (!quiz) continue;

      const selectedOption = quiz.options.find(
        opt => opt._id.toString() === userAns.optionId
      );

      const isCorrect = selectedOption ? selectedOption.isCorrect : false;

      if (isCorrect) totalScore++;

      answerDetails.push({
        questionId: quiz._id,
        selectedOption: selectedOption?.text || "",
        correct: isCorrect
      });
    }

    const quizResult = new QuizResult({
      userId,
      username,
      quizId: answers[0].quizId, // just reference
      answers: answerDetails,
      score: totalScore
    });

    await quizResult.save();

    res.json({
      message: "Quiz submitted successfully",
      score: totalScore
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Get results
router.get("/results/:userId", async (req, res) => {
  try {
    const results = await QuizResult.find({ userId: req.params.userId });
    res.json(results);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get("/leaderboard", async (req, res) => {
  try {
    const leaderboard = await QuizResult.aggregate([
      {
        $group: {
          _id: "$userId",
          username: { $first: "$username" },
          totalCorrect: { $sum: "$score" },
          totalQuestions: { $sum: { $size: "$answers" } } // 🔥 important
        }
      },
      {
        $sort: { totalCorrect: -1 }
      }
    ]);

    res.json(leaderboard);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;