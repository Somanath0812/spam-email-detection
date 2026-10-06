import { useState } from "react";
import "./App.css";

function App() {
  const [email, setEmail] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const analyzeEmail = async () => {
    if (!email.trim()) {
      setResult("Please enter an email.");
      return;
    }

    setLoading(true);
    setResult("");

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: email,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(
        data.prediction === "spam"
          ? "🚨 This email is SPAM"
          : "✅ This email is NOT SPAM"
      );
    } catch (error) {
      setResult("Backend is not ready yet. We will connect it in the next step.");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <div className="container">
        <h1>📧 Spam Email Detection</h1>

        <p className="subtitle">
          Detect whether an email is Spam or Ham using BERT AI
        </p>

        <textarea
          placeholder="Paste your email content here..."
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <button onClick={analyzeEmail} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Email"}
        </button>

        {result && <div className="result">{result}</div>}
      </div>
    </div>
  );
}

export default App;