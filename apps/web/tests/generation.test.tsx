import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GenerationWorkspace } from "@/components/generation/GenerationWorkspace";
import type { GenerationSession, GenerationStep } from "@/lib/api/client";

const calls = vi.hoisted(() => ({
  create: vi.fn(),
  get: vi.fn(),
  cancel: vi.fn(),
  remove: vi.fn(),
}));

vi.mock("@/lib/api/client", () => ({
  api: {
    createGeneration: calls.create,
    generation: calls.get,
    cancelGeneration: calls.cancel,
    deleteGeneration: calls.remove,
  },
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, _code: string, status?: number) {
      super(message);
      this.status = status;
    }
  },
}));

const step = (index: number): GenerationStep => ({
  generation_id: "generation-1",
  step_id: `generation-1:${index}`,
  step_index: index,
  prefix_token_ids: [12, 34 + index],
  prefix_sha256: `prefix-${index}`,
  decoded_prefix: "The capital is",
  decision_position: 1,
  selected_position: 2,
  vocabulary_size: 100,
  selected_token_id: 70 + index,
  tokenizer_piece: " token",
  selected_display_text: index === 0 ? " token" : " next",
  display_note: "Contextual full-sequence decode suffix.",
  decoded_continuation: " token next".slice(0, index ? 11 : 6),
  decoded_full_sequence: "The capital is token",
  selected_raw_logit: 4.2,
  selected_model_probability: 0.5,
  selected_adjusted_probability: null,
  raw_argmax_token_id: 70 + index,
  candidates: [
    {
      token_id: 70 + index,
      tokenizer_piece: " token",
      decoded_in_isolation: " token",
      raw_logit: 4.2,
      model_probability: 0.5,
      processed_logit: 4.2,
      adjusted_probability: null,
      raw_rank: 1,
      adjusted_rank: null,
      eligible: true,
    },
  ],
  distribution_statistics: {
    eligible_count: 100,
    raw_probability_sum: 1,
    adjusted_probability_sum: null,
    raw_entropy_nats: 1.1,
    adjusted_entropy_nats: null,
    displayed_adjusted_mass: null,
  },
  sampling: {
    token_index: 0,
    mode: "greedy",
    temperature: 1,
    top_k: 0,
    top_p: 1,
    display_top_k: 10,
  },
  source_trace_id: null,
  source_capture_coverage: { profile: "none" },
  capture_unavailable_reason: "capture disabled",
  forward_execution_id: `forward-${index}`,
  forward_completion_status: "completed",
  forward_duration_ms: 50,
  elapsed_ms: 60,
  provenance: {
    model_revision: "revision",
    device: "cpu",
    dtype: "float32",
    use_cache: false,
    forward_count: 1,
    rng_policy: "not used in greedy mode",
    sampling_policy: "raw-logit argmax",
  },
});

const completed = (steps: GenerationStep[]): GenerationSession => ({
  generation_id: "generation-1",
  request_key: "request-1",
  original_prompt: "The capital is",
  original_input_ids: [12, 34],
  original_decoded_prompt: "The capital is",
  model_id: "qwen3-0.6b",
  hf_model_id: "Qwen/Qwen3-0.6B",
  model_revision: "c1899de289a04d12100db370d81485cdf75e47ca",
  tokenizer_revision: "c1899de289a04d12100db370d81485cdf75e47ca",
  tokenizer_identity: "Qwen2Tokenizer",
  device: "cpu",
  dtype: "float32",
  execution_configuration: { use_cache: false },
  sampling: {
    token_index: 0,
    mode: "greedy",
    temperature: 1,
    top_k: 0,
    top_p: 1,
    display_top_k: 10,
  },
  started_at: new Date().toISOString(),
  status: "completed",
  completed_at: new Date().toISOString(),
  stop_reason: "max_new_tokens",
  max_new_tokens: steps.length,
  budgets: {},
  completed_steps: steps.length,
  attempted_forwards: steps.length,
  active_forward_execution_id: null,
  cancellation_requested: false,
  selected_token_ids: steps.map((item) => item.selected_token_id),
  decoded_continuation: " token next",
  decoded_full_sequence: "The capital is token next",
  trace_retention_summary: { original_traces: 0, available_traces: 0, retained_bytes: 0 },
  result_bytes: 2048,
  accounted_state_bytes: 4194304,
  session_duration_ms: 120,
  expires_at: new Date(Date.now() + 600000).toISOString(),
  error: null,
  steps,
} as GenerationSession);

function mount(prompt = "The capital is") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <GenerationWorkspace prompt={prompt} onOpenTrace={vi.fn()} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  calls.create.mockReset();
  calls.get.mockReset();
  calls.cancel.mockReset();
  calls.remove.mockReset();
  Object.defineProperty(globalThis, "crypto", {
    configurable: true,
    value: { randomUUID: () => "stable-test-request" },
  });
  calls.create.mockResolvedValue(completed([step(0), step(1)]));
});

describe("Generation Journey", () => {
  it("submits raw prompt IDs policy and keeps greedy probabilities distinct", async () => {
    mount();
    expect(screen.getByLabelText("Generation mode")).toHaveValue("greedy");
    expect(screen.queryByLabelText("Temperature")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /START GENERATION/i }));
    await screen.findByText("Selected token ID 71");
    const body = calls.create.mock.calls[0][0];
    expect(body.input).toEqual({ mode: "raw", text: "The capital is", add_special_tokens: false });
    expect(body.use_cache).toBe(false);
    expect(body.sampling).toMatchObject({ mode: "greedy", temperature: 1, top_k: 0, top_p: 1 });
    expect(body.sampling.seed).toBeUndefined();
    expect(screen.getByText("Not applicable in greedy mode")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Top raw model candidates" })).toBeInTheDocument();
  });

  it("validates sampled settings before any generation request", async () => {
    mount();
    await userEvent.selectOptions(screen.getByLabelText("Generation mode"), "sample");
    fireEvent.change(screen.getByLabelText("Sampling seed"), { target: { value: "-1", valueAsNumber: -1 } });
    await userEvent.click(screen.getByRole("button", { name: /START GENERATION/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Seed must be a nonnegative");
    expect(calls.create).not.toHaveBeenCalled();
  });

  it("moves through exact saved step identities using arrow navigation", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: /START GENERATION/i }));
    const first = await screen.findByRole("option", { name: /ID 70/ });
    const second = screen.getByRole("option", { name: /ID 71/ });
    expect(second).toHaveAttribute("aria-selected", "true");
    second.focus();
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "ArrowLeft" });
    await waitFor(() => expect(first).toHaveAttribute("aria-selected", "true"));
    expect(screen.getByText("Selected token ID 70")).toBeInTheDocument();
    expect(screen.getByText("Exact prefix IDs").parentElement).toHaveTextContent("12 34");
  });

  it("does not offer inspector actions when no original trace was retained", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: /START GENERATION/i }));
    await screen.findByText("Selected token ID 71");
    expect(screen.queryByRole("button", { name: "Attention Microscope" })).not.toBeInTheDocument();
    expect(screen.getByText(/no detailed step trace was captured/)).toBeInTheDocument();
  });
});
