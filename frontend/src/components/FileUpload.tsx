import React, { useState, useRef, useEffect } from "react";
import { AnalysisResult } from "../types";

const API = "http://localhost:8000/api/analyze";

interface Props {
  onResult: (result: AnalysisResult) => void;
  onAnalyzing: (v: boolean) => void;
}

const ACCEPT = ".jpg,.jpeg,.png,.webp,.mp4,.webm,.mov,.mp3,.wav,.ogg,.bmp";

export default function FileUpload({ onResult, onAnalyzing }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<string>("");
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFile = (f: File) => {
    setError("");
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    const t = f.type;
    if (t.startsWith("image")) setMediaType("image");
    else if (t.startsWith("video")) setMediaType("video");
    else if (t.startsWith("audio")) setMediaType("audio");
    else setMediaType("unknown");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    onAnalyzing(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(API, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Server error" }));
        throw new Error(err.detail || "Analysis failed");
      }
      const data: AnalysisResult = await res.json();
      onResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
      onAnalyzing(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Drop Zone */}
      <div
        onDragEnter={() => setDragging(true)}
        onDragLeave={() => setDragging(false)}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
          transition-all duration-300 select-none
          ${dragging
            ? "border-cyan-400 bg-cyan-400/10"
            : "border-white/20 hover:border-cyan-400/60 hover:bg-white/5"
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        <div className="text-5xl mb-3">📁</div>
        <p className="text-white/80 text-lg font-medium">
          Drop your file here or <span className="text-cyan-400">click to browse</span>
        </p>
        <p className="text-white/40 text-sm mt-1">
          Images (JPG, PNG, WebP) • Videos (MP4, WebM) • Audio (MP3, WAV)
        </p>
      </div>

      {/* Preview */}
      {preview && file && (
        <div className="rounded-xl overflow-hidden bg-white/5 border border-white/10 p-3">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">
              {mediaType === "image" ? "🖼️" : mediaType === "video" ? "🎬" : "🎵"}
            </span>
            <span className="text-white/80 text-sm font-mono truncate">{file.name}</span>
            <span className="ml-auto text-white/40 text-xs">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </span>
          </div>

          {mediaType === "image" && (
            <img
              src={preview}
              alt="Preview"
              className="w-full max-h-64 object-contain rounded-lg"
            />
          )}
          {mediaType === "video" && (
            <video
              src={preview}
              controls
              className="w-full max-h-64 rounded-lg"
            />
          )}
          {mediaType === "audio" && (
            <audio src={preview} controls className="w-full" />
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/40 rounded-xl p-3 text-red-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Analyze Button */}
      <button
        onClick={analyze}
        disabled={!file || loading}
        className={`
          w-full py-4 rounded-xl font-bold text-lg tracking-wide transition-all duration-300
          ${file && !loading
            ? "bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/25 hover:scale-[1.02] active:scale-[0.98]"
            : "bg-white/10 text-white/30 cursor-not-allowed"
          }
        `}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Analyzing...
          </span>
        ) : (
          "🔍 Analyze for Deepfake"
        )}
      </button>
    </div>
  );
}
