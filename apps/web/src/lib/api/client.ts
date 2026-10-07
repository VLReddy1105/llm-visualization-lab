import type { components } from "@llm-lab/shared/api";

export type ModelStatus = components["schemas"]["ModelStatus"];
export type NextTokenResponse = components["schemas"]["NextTokenResponse"];
type HealthResponse = components["schemas"]["HealthResponse"];
type NextTokenRequest = components["schemas"]["NextTokenRequest"];
type ErrorResponse = components["schemas"]["ErrorResponse"];

export const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public readonly code: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 15000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl}${path}`, {
      ...init,
      signal: controller.signal,
      cache: "no-store",
    });
    const body = await response.json();
    if (!response.ok) {
      const error = body as Partial<ErrorResponse>;
      throw new ApiError(
        error.error?.message ?? `API request failed (${response.status}).`,
        error.error?.code ?? "API_ERROR", response.status,
      );
    }
    return body as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof Error && error.name === "AbortError") {
      throw new ApiError("The API request timed out. Check the backend status.", "API_TIMEOUT");
    }
    throw new ApiError(
      `Cannot reach FastAPI at ${apiBaseUrl}. Start the backend and check its CORS configuration.`,
      "API_UNAVAILABLE",
    );
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  modelStatus: () => request<ModelStatus>("/api/model/status"),
  loadModel: () => request<ModelStatus>("/api/v1/models/qwen3-0.6b/load", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device: "auto", dtype_policy: "scientific_default", revision: null }),
  }),
  nextToken: (text: string, topK: number) => {
    const body: NextTokenRequest = {
      model_id: "qwen3-0.6b", input: { mode: "raw", text, add_special_tokens: false }, top_k: topK,
    };
    return request<NextTokenResponse>("/api/inference/next-token", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }, 120000);
  },
};
