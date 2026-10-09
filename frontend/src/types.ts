export interface ExpertResult {
  score: number;
  verdict: "FAKE" | "REAL" | "UNKNOWN";
  weight?: number;
  available?: boolean;
  details?: Record<string, unknown>;
}

export interface AnalysisResult {
  verdict: "FAKE" | "REAL";
  label: "FORGERY DETECTED" | "VERIFIED AUTHENTIC";
  confidence: number;
  confidence_pct: number;
  fake_score: number;
  media_type: "image" | "video" | "audio";
  experts: {
    image?: ExpertResult;
    frequency?: ExpertResult;
    video?: ExpertResult;
    audio?: ExpertResult;
    temporal?: ExpertResult;
  };
  threshold: number;
  processing_time_sec: number;
  filename?: string;
}
