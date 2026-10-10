import React, { useCallback, useEffect, useRef, useState } from "react";
import { AnalysisResult } from "../types";

const API = `http://${typeof window !== 'undefined' ? window.location.hostname || 'localhost' : 'localhost'}:8000/analyze`;

interface Props {
  onResult: (result: AnalysisResult) => void;
  onAnalyzing: (v: boolean) => void;
}

type CaptureMode = "photo" | "video";
type CameraFacing = "user" | "environment";

export default function CameraCapture({ onResult, onAnalyzing }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const [mode, setMode] = useState<CaptureMode>("photo");
  const [facing, setFacing] = useState<CameraFacing>("user");
  const [cameraActive, setCameraActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [capturedURL, setCapturedURL] = useState<string | null>(null);
  const [capturedType, setCapturedType] = useState<"image" | "video">("image");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const recordTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  const startCamera = useCallback(async () => {
    stopStream();
    setError("");
    try {
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: facing,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: mode === "video",
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setCapturedBlob(null);
      setCapturedURL(null);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Camera access denied";
      setError(`Camera error: ${msg}. Please allow camera permissions.`);
    }
  }, [facing, mode, stopStream]);

  // Restart camera when mode or facing changes
  useEffect(() => {
    if (cameraActive) {
      startCamera();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, facing]);

  useEffect(() => {
    return () => {
      stopStream();
      if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    };
  }, [stopStream]);

  // ── Photo Capture ──────────────────────────────────────────────────────────
  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        stopStream();
        setCapturedBlob(blob);
        setCapturedURL(URL.createObjectURL(blob));
        setCapturedType("image");
      },
      "image/jpeg",
      0.95
    );
  };

  // ── Video Recording ────────────────────────────────────────────────────────
  const startRecording = () => {
    if (!streamRef.current) return;
    chunksRef.current = [];
    setRecordDuration(0);

    const mr = new MediaRecorder(streamRef.current, {
      mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm",
    });

    mr.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      stopStream();
      setCapturedBlob(blob);
      setCapturedURL(URL.createObjectURL(blob));
      setCapturedType("video");
    };

    mr.start(250);
    mediaRecorderRef.current = mr;
    setRecording(true);

    recordTimerRef.current = setInterval(() => {
      setRecordDuration((d) => d + 1);
    }, 1000);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    if (recordTimerRef.current) {
      clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
  };

  // ── Upload & Analyze ───────────────────────────────────────────────────────
  const analyze = async () => {
    if (!capturedBlob) return;
    setLoading(true);
    setError("");
    onAnalyzing(true);
    try {
      const ext = capturedType === "image" ? "jpeg" : "webm";
      const fname = `camera_capture.${ext}`;
      const file = new File([capturedBlob], fname, { type: capturedBlob.type });
      const form = new FormData();
      form.append("file", file);
      form.append("is_camera", "true");
      const res = await fetch(API, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Server error" }));
        throw new Error(err.detail || "Analysis failed");
      }
      const data: AnalysisResult & { preview_url?: string; previewUrl?: string } = await res.json();
      if (capturedURL) {
        data.preview_url = capturedURL;
        data.previewUrl = capturedURL;
      }
      onResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
      onAnalyzing(false);
    }
  };

  const retake = () => {
    setCapturedBlob(null);
    setCapturedURL(null);
    startCamera();
  };

  const fmtDuration = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  return (
    <div className="flex flex-col gap-4">
      {/* Mode + Facing Toggle */}
      <div className="flex gap-2">
        <div className="flex bg-white/10 rounded-xl p-1 gap-1">
          {(["photo", "video"] as CaptureMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === m
                  ? "bg-cyan-500 text-white shadow"
                  : "text-white/60 hover:text-white"
              }`}
            >
              {m === "photo" ? "📷 Photo" : "🎬 Video"}
            </button>
          ))}
        </div>
        <button
          onClick={() =>
            setFacing((f) => (f === "user" ? "environment" : "user"))
          }
          className="ml-auto px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-white/70 text-sm transition-all"
          title="Flip camera"
        >
          🔄 Flip
        </button>
      </div>

      {/* Camera / Preview Area */}
      <div className="relative rounded-2xl overflow-hidden bg-black aspect-video">
        {/* Live Feed */}
        <video
          ref={videoRef}
          muted
          playsInline
          className={`w-full h-full object-cover ${capturedBlob ? "hidden" : ""}`}
        />

        {/* Captured preview */}
        {capturedURL && capturedType === "image" && (
          <img src={capturedURL} alt="Captured" className="w-full h-full object-contain" />
        )}
        {capturedURL && capturedType === "video" && (
          <video src={capturedURL} controls className="w-full h-full" />
        )}

        {/* Placeholder when not started */}
        {!cameraActive && !capturedBlob && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/40 gap-3">
            <span className="text-6xl">📷</span>
            <span className="text-sm">Click "Start Camera" to begin</span>
          </div>
        )}

        {/* Recording indicator */}
        {recording && (
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600/90 px-3 py-1 rounded-full text-white text-sm font-medium">
            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
            REC {fmtDuration(recordDuration)}
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {/* Error */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/40 rounded-xl p-3 text-red-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Control Buttons */}
      <div className="flex gap-2">
        {!capturedBlob ? (
          <>
            <button
              onClick={startCamera}
              disabled={cameraActive && mode === "photo"}
              className="flex-1 py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium transition-all disabled:opacity-50"
            >
              {cameraActive ? "🟢 Camera Active" : "🎥 Start Camera"}
            </button>

            {mode === "photo" && cameraActive && (
              <button
                onClick={capturePhoto}
                className="flex-1 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-white font-bold transition-all"
              >
                📸 Capture
              </button>
            )}

            {mode === "video" && cameraActive && !recording && (
              <button
                onClick={startRecording}
                className="flex-1 py-3 rounded-xl bg-red-500 hover:bg-red-400 text-white font-bold transition-all"
              >
                ⏺ Start Recording
              </button>
            )}

            {mode === "video" && recording && (
              <button
                onClick={stopRecording}
                className="flex-1 py-3 rounded-xl bg-orange-500 hover:bg-orange-400 text-white font-bold transition-all animate-pulse"
              >
                ⏹ Stop & Analyze
              </button>
            )}
          </>
        ) : (
          <>
            <button
              onClick={retake}
              className="flex-1 py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium transition-all"
            >
              🔄 Retake
            </button>
            <button
              onClick={analyze}
              disabled={loading}
              className={`flex-1 py-3 rounded-xl font-bold transition-all ${
                loading
                  ? "bg-white/10 text-white/30 cursor-not-allowed"
                  : "bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:scale-[1.02]"
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Analyzing...
                </span>
              ) : (
                "🔍 Analyze"
              )}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
