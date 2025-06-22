import React, { useState } from "react";
import axios from "axios";

const EmotionTester = () => {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:5001/predict", {
        text: inputText,
      });
      setResult(res.data);
    } catch (error) {
      setResult({ error: error.response?.data?.error || "Prediction failed" });
    }
  };

  return (
    <div className="p-4 border rounded shadow mt-6">
      <h2 className="text-xl font-bold mb-2">Emotion Analyzer</h2>
      <textarea
        className="w-full border p-2 mb-2"
        rows="4"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Masukkan teks untuk analisis emosi"
      />
      <button
        className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded"
        onClick={handleAnalyze}
      >
        Analisis
      </button>

      {result && (
        <div className="mt-4 p-3 border rounded bg-gray-50">
          {result.error ? (
            <p className="text-red-600 font-semibold">{result.error}</p>
          ) : (
            <>
              <p>
                <span className="font-medium">Teks:</span> {result.text}
              </p>
              <p>
                <span className="font-medium">Emosi Terdeteksi:</span>{" "}
                <span className="text-blue-600 font-bold uppercase">
                  {result.emotion}
                </span>
              </p>
              <p>
                <span className="font-medium">Keyakinan Model:</span>{" "}
                {(result.confidence * 100).toFixed(2)}%
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default EmotionTester;
