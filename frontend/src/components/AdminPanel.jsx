"use client";

import React, { useState } from "react";
import axios from "axios";

const AdminPanel = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [message, setMessage] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [report, setReport] = useState(null);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      setMessage("Silakan pilih file CSV terlebih dahulu.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const res = await axios.post(
        "http://127.0.0.1:5001/admin/upload",
        formData
      );
      setMessage(res.data.message || "Upload berhasil");
      setPreview(null);
      setMetrics(null);
      setReport(null);
      setConfusionMatrix(null);
    } catch (error) {
      setMessage(error.response?.data?.error || "Upload gagal");
    } finally {
      setLoading(false);
    }
  };

  const fetchPreview = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://127.0.0.1:5001/admin/preview");
      setPreview(res.data);
      setMessage("");
    } catch (error) {
      setMessage(
        error.response?.data?.error || "Gagal mendapatkan preview data"
      );
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    setLoading(true);
    setMessage("");
    try {
      const res = await axios.post("http://127.0.0.1:5001/admin/train");
      setMessage(res.data.message || "Training selesai");
      setMetrics({
        accuracy: res.data.accuracy,
        precision: res.data.precision,
        recall: res.data.recall,
      });
      setReport(res.data.report);
      setConfusionMatrix(res.data.confusion_matrix);
    } catch (error) {
      setMessage(error.response?.data?.error || "Gagal melakukan training.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Admin Panel</h2>

      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
        className="mb-4"
      />

      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={handleUpload}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          disabled={loading}
        >
          Upload Dataset
        </button>
        <button
          onClick={fetchPreview}
          className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          disabled={loading}
        >
          Preview Dataset
        </button>
        <button
          onClick={trainModel}
          className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
          disabled={loading}
        >
          Train Model
        </button>
      </div>

      {message && (
        <div className="mb-6 text-gray-800 bg-yellow-100 p-3 rounded">
          {message}
        </div>
      )}

      {metrics && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Hasil Evaluasi:</h3>
          <ul className="list-disc pl-5">
            <li>
              Akurasi: <strong>{metrics.accuracy}</strong>
            </li>
            <li>
              Presisi: <strong>{metrics.precision}</strong>
            </li>
            <li>
              Recall: <strong>{metrics.recall}</strong>
            </li>
          </ul>
        </div>
      )}

      {report && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-2">Classification Report</h3>
          <table className="w-full border text-sm">
            <thead>
              <tr>
                <th className="border px-2 py-1">Label</th>
                <th className="border px-2 py-1">Precision</th>
                <th className="border px-2 py-1">Recall</th>
                <th className="border px-2 py-1">F1-Score</th>
                <th className="border px-2 py-1">Support</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(report).map(([label, scores]) =>
                typeof scores === "object" ? (
                  <tr key={label}>
                    <td className="border px-2 py-1 font-medium">{label}</td>
                    <td className="border px-2 py-1">
                      {scores.precision?.toFixed(4)}
                    </td>
                    <td className="border px-2 py-1">
                      {scores.recall?.toFixed(4)}
                    </td>
                    <td className="border px-2 py-1">
                      {scores["f1-score"]?.toFixed(4)}
                    </td>
                    <td className="border px-2 py-1">{scores.support}</td>
                  </tr>
                ) : null
              )}
            </tbody>
          </table>
        </div>
      )}

      {confusionMatrix && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-2">Confusion Matrix</h3>
          <table className="border text-sm">
            <thead>
              <tr>
                <th className="border px-2 py-1 bg-gray-100">Label</th>
                {confusionMatrix.labels.map((label, i) => (
                  <th key={i} className="border px-2 py-1 bg-gray-100">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.matrix.map((row, i) => (
                <tr key={i}>
                  <td className="border px-2 py-1 font-semibold bg-gray-100">
                    {confusionMatrix.labels[i]}
                  </td>
                  {row.map((val, j) => (
                    <td key={j} className="border px-2 py-1 text-center">
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {preview && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-2">Preview Data</h3>

          <div className="mb-4">
            <h4 className="font-semibold">Head</h4>
            <table className="w-full text-sm border mb-2">
              <thead>
                <tr>
                  <th className="border px-2 py-1">Emotion</th>
                  <th className="border px-2 py-1">Full Text</th>
                </tr>
              </thead>
              <tbody>
                {preview.head.map((row, i) => (
                  <tr key={i}>
                    <td className="border px-2 py-1">{row.emotion}</td>
                    <td className="border px-2 py-1">{row.full_text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <h4 className="font-semibold">Tail</h4>
            <table className="w-full text-sm border">
              <thead>
                <tr>
                  <th className="border px-2 py-1">Emotion</th>
                  <th className="border px-2 py-1">Full Text</th>
                </tr>
              </thead>
              <tbody>
                {preview.tail.map((row, i) => (
                  <tr key={i}>
                    <td className="border px-2 py-1">{row.emotion}</td>
                    <td className="border px-2 py-1">{row.full_text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminPanel;
