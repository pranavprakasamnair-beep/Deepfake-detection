import React, { useState, useEffect } from "react";
import FileUpload from "./components/FileUpload";
import CameraCapture from "./components/CameraCapture";
import ResultsPanel from "./components/ResultsPanel";
import HistoryView from "./components/HistoryView";
import { AnalysisResult } from "./types";

type Tab = "upload" | "camera" | "history";

function NetworkIP() {
  const [info, setInfo] = useState("Loading network info...");

  useEffect(() => {
    // Show hostname-based URL for phone access
    const host = window.location.hostname;
    setInfo(`http://${host}:8080`);
  }, []);

  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-4 text-center">
      <p className="text-white/40 text-xs mb-1 uppercase tracking-widest">📱 Open on your phone</p>
      <p className="text-cyan-400 font-mono text-sm font-bold">{info}</p>
      <p className="text-white/30 text-xs mt-1">
        Make sure your phone is on the same WiFi network
      </p>
    </div>
  );
}

function LoadingOverlay() {
  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center">
      <div className="bg-gray-900 border border-white/10 rounded-2xl p-8 flex flex-col items-center gap-4 max-w-xs text-center">
        <div className="relative w-20 h-20">
          <svg
            className="animate-spin w-20 h-20 text-cyan-500"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path
              className="opacity-80"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center text-2xl">🔍</div>
        </div>
        <div>
          <p className="text-white font-bold text-lg">Analyzing Media</p>
          <p className="text-white/50 text-sm mt-1">
            Running deepfake forensics across multiple expert models...
          </p>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("upload");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const handleResult = (r: AnalysisResult) => {
    setResult(r);
    // Auto-scroll to results
    setTimeout(() => {
      document.getElementById("results")?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  const handleSelectHistoryCase = (r: AnalysisResult) => {
    setResult(r);
    setTimeout(() => {
      document.getElementById("results")?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-white">
      {analyzing && <LoadingOverlay />}

      {/* Header */}
      <div className="border-b border-white/10 bg-[#0a0f1e]/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-3xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="text-3xl">🕵️</div>
          <div>
            <h1 className="text-white font-black text-xl tracking-tight">
              Deepfake<span className="text-cyan-400">Detector</span>
            </h1>
            <p className="text-white/40 text-xs">Multimodal Forensics v2.0</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-green-400 text-xs font-medium">LIVE</span>
          </div>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-4 py-6 flex flex-col gap-6">

        {/* Tab selector */}
        <div className="flex bg-white/5 border border-white/10 rounded-2xl p-1.5 gap-1.5">
          {(
            [
              { id: "upload", label: "📁 Upload File" },
              { id: "camera", label: "📷 Camera Capture" },
              { id: "history", label: "📜 Inspection History" },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); }}
              className={`
                flex-1 py-3 rounded-xl font-semibold text-xs sm:text-sm transition-all duration-300
                ${activeTab === tab.id
                  ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20"
                  : "text-white/50 hover:text-white hover:bg-white/5"
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Active Tab Panel */}
        {activeTab === "upload" && (
          <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
            <FileUpload onResult={handleResult} onAnalyzing={setAnalyzing} />
          </div>
        )}

        {activeTab === "camera" && (
          <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
            <CameraCapture onResult={handleResult} onAnalyzing={setAnalyzing} />
          </div>
        )}

        {activeTab === "history" && (
          <HistoryView onSelectCase={handleSelectHistoryCase} />
        )}

        {/* Results */}
        {result && (
          <div id="results">
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="text-xs uppercase tracking-wider text-cyan-400 font-bold">
                Forensic Dossier
              </span>
              <button
                onClick={() => setResult(null)}
                className="text-xs text-white/40 hover:text-white transition"
              >
                Close Report ✕
              </button>
            </div>
            <ResultsPanel result={result} />
          </div>
        )}

        {/* Phone access info */}
        {activeTab === "camera" && <NetworkIP />}

        {/* Footer */}
        <div className="text-center text-white/20 text-xs py-4 border-t border-white/5">
          Multimodal Deepfake Forensics • Dual ViT + Spatial Splicing + FFT Frequency Spectrum + Persistent SQLite Audit Trail
        </div>
      </div>
    </div>
  );
}
