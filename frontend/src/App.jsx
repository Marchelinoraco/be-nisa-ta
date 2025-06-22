// src/App.js
import React from "react";
import AdminPanel from "./components/AdminPanel";
import EmotionTester from "./components/EmotionTester";

const App = () => {
  return (
    <div className="max-w-3xl mx-auto mt-10">
      <h1 className="text-2xl font-bold text-center mb-6">
        Emotion Analysis App
      </h1>
      <AdminPanel />
      <EmotionTester />
    </div>
  );
};

export default App;
