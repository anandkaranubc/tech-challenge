import { useState } from "react";

export default function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 text-gray-800 font-sans">
      <h1 className="text-4xl font-bold mb-4">Hello World 👋</h1>
      <p className="mb-6 text-lg text-gray-600">
        React 19.1.1 + Vite + Tailwind
      </p>
      <button
        onClick={() => setCount((c) => c + 1)}
        className="px-6 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition"
      >
        Count is {count}
      </button>
    </div>
  );
}
