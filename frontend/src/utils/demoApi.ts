import axios from "axios";

const DEMO_API_BASE_URL =
  process.env.NEXT_PUBLIC_DEMO_API_URL || "http://localhost:8001";

const demoApi = axios.create({
  baseURL: DEMO_API_BASE_URL,
  timeout: 180000,
});

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  active_model_path: string | null;
  tf_version: string;
  device_used: "CPU" | "GPU";
}

export interface AnalyzeRequest {
  file: File;
  modelPath?: string;
  expectedText?: string;
}

export interface AnalyzeExampleRequest {
  exampleName: string;
  modelPath?: string;
  expectedText?: string;
}

export interface AnalyzeResponse {
  predicted_text: string;
  reference_text: string | null;
  reference_source: "manual" | "align_auto" | "none";
  wer: number | null;
  cer: number | null;
  model_path_used: string;
  latency_ms: {
    preprocess: number;
    inference: number;
    total: number;
  };
  video_stats: {
    filename: string;
    size_bytes: number;
    width: number | null;
    height: number | null;
    fps: number | null;
    frame_count: number | null;
    duration_sec: number | null;
    processed_shape: number[];
  };
  preview_url: string | null;
  debug: {
    raw_timestep_indices: number[];
    raw_timestep_tokens: string[];
    raw_timestep_text: string;
  };
  device_specs: {
    cpu_model: string | null;
    cpu_physical_cores: number | null;
    cpu_logical_cores: number | null;
    ram_total_gb: number;
    ram_available_gb: number;
    gpu_names: string[];
    gpu_memory_total_mb: number | null;
    tf_version: string;
    device_used: "CPU" | "GPU";
  };
}

export async function checkDemoHealth(): Promise<HealthStatus> {
  const { data } = await demoApi.get<HealthStatus>("/health");
  return data;
}

export async function analyzeDemoVideo(
  payload: AnalyzeRequest,
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", payload.file);
  if (payload.modelPath?.trim()) {
    formData.append("model_path", payload.modelPath.trim());
  }
  if (payload.expectedText?.trim()) {
    formData.append("expected_text", payload.expectedText.trim());
  }

  const { data } = await demoApi.post<AnalyzeResponse>("/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export interface ExampleListResponse {
  base_dir: string;
  count: number;
  examples: string[];
}

export async function listDemoExamples(
  limit: number = 100,
): Promise<ExampleListResponse> {
  const { data } = await demoApi.get<ExampleListResponse>("/examples", {
    params: { limit },
  });
  return data;
}

export async function analyzeDemoExample(
  payload: AnalyzeExampleRequest,
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("example_name", payload.exampleName);
  if (payload.modelPath?.trim()) {
    formData.append("model_path", payload.modelPath.trim());
  }
  if (payload.expectedText?.trim()) {
    formData.append("expected_text", payload.expectedText.trim());
  }

  const { data } = await demoApi.post<AnalyzeResponse>(
    "/analyze-example",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
    },
  );
  return data;
}

export default demoApi;
