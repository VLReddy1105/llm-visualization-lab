import type { components } from "@llm-lab/shared/api";

export type TokenMetadata = components["schemas"]["TokenMetadata"];
export type Candidate = components["schemas"]["Candidate"];
export type ApiPage = components["schemas"]["Page"];
export type ModelStatus = components["schemas"]["ModelStatus"];
export type NextTokenResponse = components["schemas"]["NextTokenResponse"];
export type AnalysisJob = components["schemas"]["AnalysisJob"];
export type AnalysisRequest = AnalysisJob["request"];
export type EvolutionResult = components["schemas"]["EvolutionResult"];
export type LensResult = components["schemas"]["LensResult"];
export type SamplingResult = components["schemas"]["SamplingResult"];
export type SamplingParameters = components["schemas"]["SamplingParameters"];
export type GenerationRequest = components["schemas"]["GenerationRequest"];
export type GenerationSession = components["schemas"]["GenerationSession"];
export type GenerationStep = components["schemas"]["GenerationStep"];
export type GenerationSampling = components["schemas"]["GenerationSampling"];
export type BlockParameters = components["schemas"]["BlockParameters"];
export type ResidualStreamResult = components["schemas"]["ResidualStreamResult"];
export type MLPInternalsResult = components["schemas"]["MLPInternalsResult"];
export type Checkpoint = Pick<
  EvolutionResult["checkpoints"][number],
  "checkpoint_index" | "component_id" | "label" | "stage" | "layer_index"
>;
type HealthResponse = components["schemas"]["HealthResponse"];
type NextTokenRequest = components["schemas"]["NextTokenRequest"];
export type CreateTraceRequest = components["schemas"]["CreateTraceRequest"];
export type TraceEnvelope = components["schemas"]["TraceEnvelope"];
export type TraceOverview = components["schemas"]["TraceOverview"];
export type ComponentDescriptor = components["schemas"]["ComponentDescriptor"];
export type ComponentGroup = components["schemas"]["ComponentGroup"];
export type TensorSlice = components["schemas"]["TensorSlice"];
export type SliceRequest = components["schemas"]["SliceRequest"];
export type AttentionHead = components["schemas"]["AttentionHead"];
export type AttentionRelation = components["schemas"]["AttentionRelation"];
export type OutputComparison = components["schemas"]["OutputComparison"];
export type TraceLogits = components["schemas"]["LogitsResponse"];
export type TraceStoreStatus = components["schemas"]["StoreStatus"];
export type TokenPage = components["schemas"]["TokenPage"];
export type ComponentPage = components["schemas"]["ComponentPage"];
type Page = components["schemas"]["Page"];
type CancellationResponse = components["schemas"]["CancellationResponse"];
type JsonSliceRequest = Omit<SliceRequest, "format" | "include_statistics"> & {
  format?: "json";
  include_statistics?: boolean;
};
export type BinaryTensorSlice = {
  data: ArrayBuffer;
  metadata: Omit<
    TensorSlice,
    "values" | "nonfinite" | "masked_flat_indices" | "wire_dtype"
  > & {
    wire_dtype: "little_endian_float32";
    nonfinite_encoding: string;
  };
};
type PageOptions = { limit?: number; cursor?: string };
type HeadOptions = {
  query_token_index?: number;
  include_weights?: boolean;
  include_query?: boolean;
};
type ErrorResponse = components["schemas"]["ErrorResponse"];

export const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly status?: number,
    public readonly details?: ErrorResponse["error"]["details"],
    public readonly retryable = false,
    public readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 15000,
  binary = false,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl}${path}`, {
      ...init,
      signal: init?.signal
        ? AbortSignal.any([controller.signal, init.signal])
        : controller.signal,
      cache: "no-store",
    });
    if (response.status === 204) return undefined as T;
    if (response.ok && binary) {
      const header = response.headers.get("X-Tensor-Metadata");
      if (!header)
        throw new ApiError(
          "Binary slice metadata is missing.",
          "INVALID_RESPONSE",
        );
      return {
        data: await response.arrayBuffer(),
        metadata: JSON.parse(header),
      } as T;
    }
    const body = await response.json();
    if (!response.ok) {
      const error = body as Partial<ErrorResponse>;
      throw new ApiError(
        error.error?.message ?? `API request failed (${response.status}).`,
        error.error?.code ?? "API_ERROR",
        response.status,
        error.error?.details,
        error.error?.retryable,
        error.error?.request_id,
      );
    }
    return body as T;
  } catch (error) {
    if (init?.signal?.aborted) throw error;
    if (error instanceof ApiError) throw error;
    if (error instanceof Error && error.name === "AbortError") {
      throw new ApiError(
        "The API request timed out. Check the backend status.",
        "API_TIMEOUT",
      );
    }
    throw new ApiError(
      `Cannot reach FastAPI at ${apiBaseUrl}. Start the backend and check its CORS configuration.`,
      "API_UNAVAILABLE",
    );
  } finally {
    clearTimeout(timer);
  }
}

function query(
  path: string,
  options: Record<string, string | number | boolean | undefined> = {},
): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(options)) {
    if (value !== undefined) params.set(key, String(value));
  }
  return params.size ? `${path}?${params}` : path;
}

const tracePath = (id: string) => `/api/v1/traces/${encodeURIComponent(id)}`;
const generationPath = (id: string) =>
  `/api/v1/generations/${encodeURIComponent(id)}`;
const layerPath = (id: string, layer: number) =>
  `${tracePath(id)}/layers/${layer}`;
const componentPath = (id: string, component: string) =>
  `${tracePath(id)}/components/${encodeURIComponent(component)}`;
const jsonPost = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  modelStatus: (signal?: AbortSignal) =>
    request<ModelStatus>("/api/model/status", { signal }),
  loadModel: () =>
    request<ModelStatus>("/api/v1/models/qwen3-0.6b/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device: "auto",
        dtype_policy: "scientific_default",
        revision: null,
      }),
    }),
  nextToken: (text: string, topK: number) => {
    const body: NextTokenRequest = {
      model_id: "qwen3-0.6b",
      input: { mode: "raw", text, add_special_tokens: false },
      top_k: topK,
    };
    return request<NextTokenResponse>(
      "/api/inference/next-token",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      120000,
    );
  },
  createTrace: (body: CreateTraceRequest, wait = false, signal?: AbortSignal) =>
    request<TraceEnvelope>(
      query("/api/v1/traces", { wait }),
      { ...jsonPost(body), signal },
      120000,
    ),
  trace: (id: string, displayTopK = 10, signal?: AbortSignal) =>
    request<TraceEnvelope>(
      query(tracePath(id), { display_top_k: displayTopK }),
      { signal },
    ),
  traceStatus: () => request<TraceStoreStatus>("/api/v1/traces/status"),
  listTraces: (options: PageOptions = {}) =>
    request<Page>(query("/api/v1/traces", options)),
  traceTokens: (id: string, options: PageOptions = {}, signal?: AbortSignal) =>
    request<TokenPage>(query(`${tracePath(id)}/tokens`, options), { signal }),
  traceLayers: (id: string, options: PageOptions = {}, signal?: AbortSignal) =>
    request<Page>(query(`${tracePath(id)}/layers`, options), { signal }),
  traceLayer: (id: string, layer: number, signal?: AbortSignal) =>
    request<ComponentGroup>(layerPath(id, layer), { signal }),
  compareOutputs: (id: string, reference: string, signal?: AbortSignal) =>
    request<OutputComparison>(
      `${tracePath(id)}/compare-output/${encodeURIComponent(reference)}`,
      { signal },
    ),
  traceAttention: (id: string, layer: number, signal?: AbortSignal) =>
    request<ComponentGroup>(`${layerPath(id, layer)}/attention`, { signal }),
  traceHead: (
    id: string,
    layer: number,
    head: number,
    options: HeadOptions = {},
    signal?: AbortSignal,
  ) =>
    request<AttentionHead>(
      query(`${layerPath(id, layer)}/attention/heads/${head}`, options),
      { signal },
    ),
  traceRelation: (
    id: string,
    layer: number,
    head: number,
    queryToken: number,
    keyToken: number,
    signal?: AbortSignal,
  ) =>
    request<AttentionRelation>(
      query(`${layerPath(id, layer)}/attention/heads/${head}/relation`, {
        query_token_index: queryToken,
        key_token_index: keyToken,
      }),
      { signal },
    ),
  traceMLP: (id: string, layer: number) =>
    request<ComponentGroup>(`${layerPath(id, layer)}/mlp`),
  traceResidual: (id: string, layer: number) =>
    request<ComponentGroup>(`${layerPath(id, layer)}/residual`),
  traceLogits: (
    id: string,
    displayTopK = 10,
    tokenIndex?: number,
    signal?: AbortSignal,
  ) =>
    request<TraceLogits>(
      query(`${tracePath(id)}/logits`, {
        display_top_k: displayTopK,
        token_index: tokenIndex,
      }),
      { signal },
    ),
  traceComponents: (id: string, options: PageOptions = {}) =>
    request<ComponentPage>(query(`${tracePath(id)}/components`, options)),
  traceComponent: (id: string, component: string, signal?: AbortSignal) =>
    request<ComponentDescriptor>(componentPath(id, component), { signal }),
  traceSlice: (
    id: string,
    component: string,
    body: JsonSliceRequest,
    signal?: AbortSignal,
  ) =>
    request<TensorSlice>(`${componentPath(id, component)}/slice`, {
      ...jsonPost(body),
      signal,
    }),
  traceTensorSlice: (id: string, tensor: string, body: JsonSliceRequest) =>
    request<TensorSlice>(
      `${tracePath(id)}/tensors/${encodeURIComponent(tensor)}/slice`,
      jsonPost(body),
    ),
  traceBinarySlice: (
    id: string,
    component: string,
    body: Omit<SliceRequest, "format">,
  ) =>
    request<BinaryTensorSlice>(
      `${componentPath(id, component)}/slice`,
      jsonPost({ ...body, format: "binary" }),
      15000,
      true,
    ),
  traceBinaryTensorSlice: (
    id: string,
    tensor: string,
    body: Omit<SliceRequest, "format">,
  ) =>
    request<BinaryTensorSlice>(
      `${tracePath(id)}/tensors/${encodeURIComponent(tensor)}/slice`,
      jsonPost({ ...body, format: "binary" }),
      15000,
      true,
    ),
  createAnalysis: (id: string, body: AnalysisRequest, signal?: AbortSignal) =>
    request<AnalysisJob>(`${tracePath(id)}/analyses`, {
      ...jsonPost(body),
      signal,
    }),
  analysis: (id: string, analysisId: string, signal?: AbortSignal) =>
    request<AnalysisJob>(
      `${tracePath(id)}/analyses/${encodeURIComponent(analysisId)}`,
      { signal },
    ),
  cancelAnalysis: (id: string, analysisId: string) =>
    request<AnalysisJob>(
      `${tracePath(id)}/analyses/${encodeURIComponent(analysisId)}`,
      { method: "DELETE" },
    ),
  deleteTrace: (id: string) =>
    request<CancellationResponse | undefined>(tracePath(id), {
      method: "DELETE",
    }),
  createGeneration: (body: GenerationRequest, signal?: AbortSignal) =>
    request<GenerationSession>(
      "/api/v1/generations",
      { ...jsonPost(body), signal },
      15000,
    ),
  generation: (id: string, signal?: AbortSignal) =>
    request<GenerationSession>(generationPath(id), { signal }),
  generationStep: (id: string, index: number, signal?: AbortSignal) =>
    request<GenerationStep>(
      `${generationPath(id)}/steps/${index}`,
      { signal },
    ),
  cancelGeneration: (id: string) =>
    request<GenerationSession>(`${generationPath(id)}/cancel`, {
      method: "POST",
    }),
  deleteGeneration: (id: string) =>
    request<void>(generationPath(id), { method: "DELETE" }),
};
