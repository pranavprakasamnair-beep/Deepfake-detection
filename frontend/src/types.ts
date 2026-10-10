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
  case_number?: string;
  inspection_count?: number;
  media_path?: string;
  file_type?: string;
}

export interface HistoryCase {
  file_hash: string;
  case_number: string;
  filename: string;
  media_type: "image" | "video" | "audio";
  verdict: "FAKE" | "REAL";
  confidence: number;
  first_inspected: string;
  last_inspected: string;
  inspection_count: number;
  media_path?: string;
  details?: Record<string, any>;
}
