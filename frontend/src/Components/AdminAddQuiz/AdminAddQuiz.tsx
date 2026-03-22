import React, { useState, useEffect } from "react";
import axios from "axios";
import "./AdminAddQuiz.css";
import { toastError, toastSucces } from "../../Model/toast";

interface Option {
  text: string;
  isCorrect: boolean;
}

interface Quiz {
  _id: string;
  question: string;
  options: Option[];
  createdBy: string;
}

const AdminQuizPage: React.FC = () => {
  const [question, setQuestion] = useState<string>("");
  const [options, setOptions] = useState<Option[]>([{ text: "", isCorrect: false }]);
  const [quizzes, setQuizzes] = useState<Quiz[]>([]);

  // Fetch all quizzes added by this admin
  const fetchQuizzes = async () => {
    try {
      const storedUser = localStorage.getItem("user");
      const userObj = storedUser ? JSON.parse(storedUser) : null;

      const res = await axios.get('http://localhost:3000/api/quiz/all');
      setQuizzes(res.data);
    } catch (err) {
      console.error(err);
      toastError("Error fetching quizzes");
    }
  };

  useEffect(() => {
    fetchQuizzes();
  }, []);

  const addOption = () => setOptions([...options, { text: "", isCorrect: false }]);

  const handleOptionChange = (index: number, field: keyof Option, value: string | boolean) => {
    setOptions((prevOptions) => {
      const updated = [...prevOptions];
      if (field === "text" && typeof value === "string") updated[index].text = value;
      else if (field === "isCorrect" && typeof value === "boolean") updated[index].isCorrect = value;
      return updated;
    });
  };

  const submitQuiz = async () => {
    try {
      const storedUser = localStorage.getItem("user");
      const userObj = storedUser ? JSON.parse(storedUser) : null;

      await axios.post("http://localhost:3000/api/quiz/admin/add", {
        question,
        options,
        createdBy: userObj?.username,
      });

      toastSucces("Quiz added successfully!");
      setQuestion("");
      setOptions([{ text: "", isCorrect: false }]);
      fetchQuizzes(); // Refresh quiz list
    } catch (err) {
      console.error(err);
      toastError("Error adding quiz");
    }
  };

  return (
    <div className="admin-quiz-page">
      <h2>Add New Quiz</h2>

      <input
        className="quiz-question-input"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Enter Question"
      />

      <div className="options-container">
        {options.map((opt, idx) => (
          <div className="option-card" key={idx}>
            <input
              className="option-text-input"
              value={opt.text}
              onChange={(e) => handleOptionChange(idx, "text", e.target.value)}
              placeholder={`Option ${idx + 1}`}
            />
            <label className="correct-label">
              <input
                type="checkbox"
                checked={opt.isCorrect}
                onChange={(e) => handleOptionChange(idx, "isCorrect", e.target.checked)}
              />
              Correct
            </label>
          </div>
        ))}
      </div>

      <div className="buttons-container">
        <button className="add-option-btn" onClick={addOption}>
          Add Option
        </button>
        <button className="submit-quiz-btn" onClick={submitQuiz}>
          Submit Quiz
        </button>
      </div>

      <hr />

      <h2>All Quizzes Added by You</h2>
      {quizzes.length === 0 ? (
        <p>No quizzes added yet.</p>
      ) : (
        <div className="quiz-list">
          {quizzes.map((quiz) => (
            <div key={quiz._id} className="quiz-card">
              <p className="quiz-question">{quiz.question}</p>
              <ul className="quiz-options">
                {quiz.options.map((opt, idx) => (
                  <li key={idx} className={opt.isCorrect ? "correct-option" : ""}>
                    {opt.text} {opt.isCorrect ? "(Correct)" : ""}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AdminQuizPage;