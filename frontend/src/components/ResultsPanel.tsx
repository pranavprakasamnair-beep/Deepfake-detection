import React from "react";
import { AnalysisResult, ExpertResult } from "../types";

interface Props {
  result: AnalysisResult;
}

const EXPERT_LABELS: Record<string, { label: string; icon: string; description: string }> = {
  image:     { label: "Image AI Model",     icon: "🧠", description: "EfficientNet-B4 + ViT ensemble" },
  frequency: { label: "Frequency Analysis", icon: "📡", description: "FFT + DCT artifact detection" },
  video:     { label: "Video Analysis",     icon: "🎬", description: "Frame-level deepfake scoring" },
  audio:     { label: "Audio Analysis",     icon: "🎙️", description: "MFCC + SVM voice forensics" },
  temporal:  { label: "Temporal Analysis",  icon: "⏱️", description: "Inter-frame motion consistency" },
};

function ExpertBar({
  expertKey,
  expert,
}: {
  expertKey: string;
  expert: ExpertResult;
}) {
  const info = EXPERT_LABELS[expertKey] || {
    label: expertKey,
    icon: "🔍",
    description: "",
  };
  const pct = Math.round(expert.score * 100);
  const isFake = expert.verdict === "FAKE";
  const isUnknown = expert.verdict === "UNKNOWN";

  return (
    <div className="bg-white/5 rounded-xl p-4 border border-white/10">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xl">{info.icon}</span>
          <div>
            <div className="text-white font-medium text-sm">{info.label}</div>
            <div className="text-white/40 text-xs">{info.description}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`text-xs font-bold px-2 py-1 rounded-lg ${
              isUnknown
                ? "bg-gray-500/30 text-gray-300"
                : isFake
                ? "bg-red-500/30 text-red-300"
                : "bg-green-500/30 text-green-300"
            }`}
          >
            {isUnknown ? "N/A" : isFake ? "FAKE" : "REAL"}
          </span>
          <span className="text-white font-mono text-sm font-bold">
            {isUnknown ? "--" : `${pct}%`}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        {!isUnknown && (
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              isFake
                ? "bg-gradient-to-r from-orange-500 to-red-500"
                : "bg-gradient-to-r from-cyan-500 to-green-500"
            }`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      {!isUnknown && (
        <div className="flex justify-between mt-1">
          <span className="text-white/30 text-xs">REAL ←</span>
          <span className="text-white/30 text-xs">→ FAKE</span>
        </div>
      )}
    </div>
  );
}

export default function ResultsPanel({ result }: Props) {
  const isFake = result.verdict === "FAKE";
  const confPct = result.confidence_pct;

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      {/* Main Verdict Banner */}
      <div
        className={`relative rounded-2xl p-6 border-2 overflow-hidden ${
          isFake
            ? "border-red-500/60 bg-red-950/40"
            : "border-green-500/60 bg-green-950/40"
        }`}
      >
        {/* Glow */}
        <div
          className={`absolute inset-0 ${
            isFake
              ? "bg-gradient-to-br from-red-900/30 to-transparent"
              : "bg-gradient-to-br from-green-900/30 to-transparent"
          }`}
        />
        <div className="relative z-10">
          <div className="flex items-center gap-4 mb-4">
            <div className="text-5xl">{isFake ? "🚨" : "✅"}</div>
            <div>
              <div
                className={`text-3xl font-black tracking-tight ${
                  isFake ? "text-red-400" : "text-green-400"
                }`}
              >
                {result.label}
              </div>
              <div className="text-white/60 text-sm mt-1">
                Media type:{" "}
                <span className="text-white/80 capitalize">{result.media_type}</span>
                {result.filename && (
                  <> &nbsp;•&nbsp; <span className="font-mono text-white/60">{result.filename}</span></>
                )}
              </div>
            </div>
          </div>

          {/* Confidence bar */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-white/60 text-sm">Detection Confidence</span>
              <span
                className={`font-black text-2xl ${
                  isFake ? "text-red-400" : "text-green-400"
                }`}
              >
                {confPct}%
              </span>
            </div>
            <div className="h-4 bg-black/40 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-1000 ${
                  isFake
                    ? "bg-gradient-to-r from-orange-500 to-red-500 shadow-lg shadow-red-500/50"
                    : "bg-gradient-to-r from-cyan-500 to-green-500 shadow-lg shadow-green-500/50"
                }`}
                style={{ width: `${confPct}%` }}
              />
            </div>
          </div>

          {/* Fake probability score */}
          <div className="mt-3 text-white/40 text-xs">
            Raw fake score: {(result.fake_score * 100).toFixed(1)}% &nbsp;•&nbsp; Threshold:{" "}
            {(result.threshold * 100).toFixed(0)}% &nbsp;•&nbsp; Analysis time:{" "}
            {result.processing_time_sec}s
          </div>
        </div>
      </div>

      {/* Expert Breakdown */}
      <div>
        <h3 className="text-white/60 text-sm font-medium uppercase tracking-widest mb-3">
          Expert Analysis Breakdown
        </h3>
        <div className="flex flex-col gap-3">
          {Object.entries(result.experts).map(([key, expert]) => (
            <ExpertBar key={key} expertKey={key} expert={expert} />
          ))}
        </div>
      </div>

      {/* Interpretation note */}
      <div className="bg-white/5 border border-white/10 rounded-xl p-4 text-white/50 text-sm">
        <span className="text-white/70 font-medium">ℹ️ How to read this: </span>
        Each expert bar shows the probability of deepfake content detected by that analysis method.
        Higher % = more suspicious. The final verdict combines all experts using weighted fusion.
      </div>
    </div>
  );
}
