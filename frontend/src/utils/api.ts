import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export interface AnalyzeResult {
  text: string;
  processing_time: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
}

/** POST /analyze — upload a video file for lip-reading inference */
export async function analyzeVideo(file: File): Promise<AnalyzeResult> {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await api.post<AnalyzeResult>("/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/** GET /health — check if backend and model are ready */
export async function checkHealth(): Promise<HealthStatus> {
  const { data } = await api.get<HealthStatus>("/health");
  return data;
}

export default api;
