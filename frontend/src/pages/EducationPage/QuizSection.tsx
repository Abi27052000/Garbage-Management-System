

// UserQuizPage.tsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import "./QuizSection.css";
import { toastError, toastSucces } from "../../Model/toast";
import { updateUserPoints, users as allUsers } from "../../utility/data";

interface Option {
  _id: string;
  text: string;
  isCorrect: boolean;
}

interface Quiz {
  _id: string;
  question: string;
  options: Option[];
}

interface UserAnswer {
  quizId: string;
  selectedOptionId: string | null;
}

const UserQuizPage: React.FC = () => {
  const [quizzes, setQuizzes] = useState<Quiz[]>([]);
  const [answers, setAnswers] = useState<UserAnswer[]>([]);
  const [result, setResult] = useState<number | null>(null);

  // Fetch quizzes from backend
  useEffect(() => {
    const fetchQuizzes = async () => {
      try {
        const res = await axios.get("http://localhost:3000/api/quiz/all"); // your backend endpoint
        setQuizzes(res.data);

        // initialize answers
        setAnswers(res.data.map((q: Quiz) => ({ quizId: q._id, selectedOptionId: null })));
      } catch (err) {
        console.error(err);
        toastError("Error fetching quizzes");
      }
    };
    fetchQuizzes();
  }, []);

  const handleSelectOption = (quizId: string, optionId: string) => {
    setAnswers((prev) =>
      prev.map((ans) =>
        ans.quizId === quizId ? { ...ans, selectedOptionId: optionId } : ans
      )
    );
  };

  const submitAnswers = async () => {
  try {
    const storedUser = localStorage.getItem("user");
    const userObj = storedUser ? JSON.parse(storedUser) : null;

    if (!userObj) {
      toastError("User not logged in");
      return;
    }

    const formattedAnswers = answers
      .filter(a => a.selectedOptionId)
      .map(a => ({
        quizId: a.quizId,
        optionId: a.selectedOptionId
      }));

    if (formattedAnswers.length === 0) {
      toastError("Please answer at least one question");
      return;
    }

    const res = await axios.post("http://localhost:3000/api/quiz/submit", {
      userId: userObj.id,
      username: userObj.username,
      answers: formattedAnswers
    });

    setResult(res.data.score);

    toastSucces(`Quiz submitted! Score: ${res.data.score}`);

  } catch (err) {
    console.error(err);
    toastError("Error submitting quiz");
  }
};

  return (
    <div className="user-quiz-page">
      <h2>Waste Management Quiz 🌱</h2>

      {quizzes.length === 0 ? (
        <p>No quizzes available at the moment.</p>
      ) : (
        quizzes.map((quiz, qIndex) => (
          <div key={quiz._id} className="quiz-card">
            <p className="quiz-question">
              {qIndex + 1}. {quiz.question}
            </p>
            <div className="quiz-options">
              {quiz.options.map((opt) => (
                <label key={opt._id} className="option-label">
                  <input
                    type="radio"
                    name={quiz._id}
                    checked={answers[qIndex]?.selectedOptionId === opt._id}
                    onChange={() => handleSelectOption(quiz._id, opt._id)}
                  />
                  {opt.text}
                </label>
              ))}
            </div>
          </div>
        ))
      )}

      {quizzes.length > 0 && (
        <button className="submit-quiz-btn" onClick={submitAnswers}>
          Submit Quiz
        </button>
      )}

      {result !== null && (
        <div className="quiz-result">
          <h3>
            Your Score: {result} / {quizzes.length}
          </h3>
        </div>
      )}
    </div>
  );
};

export default UserQuizPage;