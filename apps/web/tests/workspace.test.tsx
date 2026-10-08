import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
  act,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LLMWorkspace } from "@/components/workspace/LLMWorkspace";
import { createSelectionStore } from "@/state/selection";
import { api, type TraceEnvelope } from "@/lib/api/client";

// These are recorded real Qwen HTTP responses, used only for isolated HTTP tests.
function evidence(name: string) {
  return JSON.parse(
    readFileSync(resolve(process.cwd(), "../../docs/validation", name), "utf8"),
  );
}
const recorded: TraceEnvelope = evidence("stage4-trace-overview.json");
const tokens = evidence("stage4-tokens.json");
const layers = evidence("stage4-layers.json");
const layer = evidence("stage4-layer12.json");
const logits = evidence("stage4-logits.json");
const model = {
  state: "ready",
  model_id: "qwen3-0.6b",
  hf_model_id: recorded.overview!.model.hf_model_id,
  device: recorded.overview!.model.device,
  dtype: recorded.overview!.model.dtype,
};
let current: TraceEnvelope;
let getOverview: () => TraceEnvelope;
let createResponse: () => Response;
type FetchFn = (input: string, init?: RequestInit) => Promise<Response>;
let fetchMock: ReturnType<typeof vi.fn<FetchFn>>;
const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
const err = (code: string, message: string, status = 409) =>
  json(
    {
      error: {
        code,
        message,
        details: { hint: "Inspect the capture plan" },
        retryable: false,
        request_id: "isolated-test",
      },
    },
    status,
  );

beforeEach(() => {
  current = structuredClone(recorded);
  current.created_at = new Date().toISOString();
  current.expires_at = new Date(Date.now() + 1800000).toISOString();
  getOverview = () => current;
  createResponse = () =>
    json({ ...current, status: "running", overview: null }, 202);
  fetchMock = vi.fn(async (input: string, init?: RequestInit) => {
    const path = new URL(input).pathname;
    if (path === "/api/model/status") return json(model);
    if (path === "/api/v1/traces" && init?.method === "POST")
      return createResponse();
    if (path.endsWith("/tokens")) return json(tokens);
    if (path.endsWith("/layers")) return json(layers);
    if (/\/layers\/\d+$/.test(path)) return json(layer);
    if (path.endsWith("/logits")) return json(logits);
    if (path.endsWith("/components/embedding.output"))
      return json(evidence("stage4-embedding.json"));
    if (path.endsWith("/components/output.normalized"))
      return json(evidence("stage4-final-norm.json"));
    if (path.endsWith("/components/output.lm_head_input"))
      return json(evidence("stage4-lm-head.json"));
    if (init?.method === "DELETE") return new Response(null, { status: 204 });
    return json(getOverview());
  });
  vi.stubGlobal("fetch", fetchMock);
});
async function run() {
  render(<LLMWorkspace />);
  const button = screen.getByRole("button", { name: /TRACE MODEL/ });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  await screen.findByText("Trace ready");
  await screen.findByRole("button", { name: 'Token 3: " Germany"' });
}
function posts() {
  return fetchMock.mock.calls.filter(
    ([, init]) =>
      init?.method === "POST" && !String(init?.body).includes("selectors"),
  );
}

describe("Stage 4 workspace", () => {
  it("starts with the raw example and no model results", async () => {
    render(<LLMWorkspace />);
    expect(screen.getByLabelText("Raw prompt")).toHaveValue(
      "The capital of Germany is",
    );
    expect(
      screen.queryByRole("button", { name: "Layer 12" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText("Layer nodes appear from your trace"),
    ).toBeInTheDocument();
    await screen.findByText("Model ready");
    expect(posts()).toHaveLength(0);
  });
  it("submits exact raw text once using overview capture", async () => {
    render(<LLMWorkspace />);
    await screen.findByText("Model ready");
    fireEvent.change(screen.getByLabelText("Raw prompt"), {
      target: { value: "  test\ntext  " },
    });
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await screen.findByText("Trace ready");
    expect(posts()).toHaveLength(1);
    expect(JSON.parse(String(posts()[0][1]?.body))).toEqual({
      prompt: "  test\ntext  ",
      capture: { profile: "overview" },
      display_top_k: 10,
    });
  });
  it("shows submitting and actual running states before readiness", async () => {
    let release: (response: Response) => void = () => {};
    const normal = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((url, init) =>
      init?.method === "POST"
        ? new Promise<Response>((resolve) => {
            release = resolve;
          })
        : normal(url, init),
    );
    render(<LLMWorkspace />);
    await screen.findByText("Model ready");
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    expect(screen.getByText("Submitting")).toBeInTheDocument();
    getOverview = () => ({ ...current, status: "running", overview: null });
    await act(async () =>
      release(json({ ...current, status: "running", overview: null }, 202)),
    );
    await screen.findByText("Creating trace");
    getOverview = () => current;
    await screen.findByText("Trace ready");
    expect(posts()).toHaveLength(1);
  });
  it("renders all 28 layer buttons and the connected output stages", async () => {
    await run();
    expect(screen.getAllByRole("button", { name: /^Layer \d+$/ })).toHaveLength(
      28,
    );
    for (const name of [
      "Input",
      "Tokens",
      "Embedding",
      "Final Norm",
      "LM Head",
      "Logits",
    ])
      expect(screen.getAllByRole("button", { name }).length).toBeGreaterThan(0);
  });
  it("shows five real tokens with whitespace intact", async () => {
    await run();
    const strip = screen.getByRole("region", { name: "Input tokens" });
    expect(within(strip).getAllByRole("button")).toHaveLength(5);
    expect(screen.getByText("·Germany")).toBeInTheDocument();
  });
  it("keeps token selection when selecting Layer 12 and reads its endpoint", async () => {
    await run();
    const token = screen.getByRole("button", { name: 'Token 3: " Germany"' });
    await userEvent.click(token);
    expect(token).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("heading", { name: "Token 3" }),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Layer 12" }));
    await screen.findByText("Capture coverage");
    expect(token).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("heading", { name: "Layer 12" }),
    ).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(([url]) => url.endsWith("/layers/12")),
    ).toBe(true);
    expect(posts()).toHaveLength(1);
  });
  it("switches inspectors for embedding, norm, head and logits", async () => {
    await run();
    for (const [name, text] of [
      ["Embedding", "Maps each exact token ID"],
      ["Final Norm", "Normalizes the final residual"],
      ["LM Head", "Projects the final hidden"],
      ["Logits", "Real next-token candidates"],
    ]) {
      await userEvent.click(screen.getAllByRole("button", { name })[0]);
      await screen.findByText(new RegExp(text));
    }
    expect(
      screen.getByText("47.70%", { selector: ".prediction-summary > span" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(logits.top_k[0].raw_logit.toFixed(4)),
    ).toBeInTheDocument();
  });
  it("maintains distinct candidate and input token selection", async () => {
    await run();
    await userEvent.click(
      screen.getByRole("button", { name: 'Token 3: " Germany"' }),
    );
    await userEvent.click(screen.getAllByRole("button", { name: "Logits" })[0]);
    const candidates = await within(
      screen.getByRole("region", { name: "Logits" }),
    ).findByRole("button", { name: "·Berlin" });
    await userEvent.click(candidates);
    expect(candidates).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("button", { name: 'Token 3: " Germany"' }),
    ).toHaveAttribute("aria-pressed", "true");
  });
  it("shows uncaptured attention and MLP without inference", async () => {
    await run();
    await userEvent.click(screen.getByRole("button", { name: "Layer 12" }));
    await screen.findByText("Capture coverage");
    expect(screen.getAllByText("Not captured in this trace")).toHaveLength(2);
    expect(posts()).toHaveLength(1);
  });
  it("does not retrace edits and resets selections on explicit execution", async () => {
    await run();
    await userEvent.click(
      screen.getByRole("button", { name: 'Token 3: " Germany"' }),
    );
    fireEvent.change(screen.getByLabelText("Raw prompt"), {
      target: { value: "A different raw prompt" },
    });
    expect(screen.getByText(/Prompt changed/)).toBeInTheDocument();
    expect(posts()).toHaveLength(1);
    current = { ...current, trace_id: "new-trace" };
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await waitFor(() => expect(posts()).toHaveLength(2));
    await screen.findByText("Trace ready");
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: 'Token 3: " Germany"' }),
      ).toHaveAttribute("aria-pressed", "false"),
    );
    expect(
      within(screen.getByRole("region", { name: "Input tokens" })).getAllByRole(
        "button",
      ),
    ).toHaveLength(5);
  });
  it("caches selected layer reads and stops ready polling", async () => {
    await run();
    const select = async () =>
      userEvent.click(screen.getByRole("button", { name: "Layer 12" }));
    await select();
    await screen.findByText("Capture coverage");
    await userEvent.click(screen.getAllByRole("button", { name: "Input" })[0]);
    await select();
    await new Promise((resolve) => setTimeout(resolve, 1100));
    expect(
      fetchMock.mock.calls.filter(([url]) => url.endsWith("/layers/12")),
    ).toHaveLength(1);
    expect(
      fetchMock.mock.calls.filter(([url]) =>
        new URL(url).pathname.endsWith(current.trace_id),
      ),
    ).toHaveLength(1);
  });
  it.each([
    "CAPTURE_BUDGET_EXCEEDED",
    "TRACE_STORE_BUDGET_EXCEEDED",
    "MODEL_NOT_READY",
    "INVALID_REQUEST",
  ])("displays structured %s errors", async (code) => {
    createResponse = () => err(code, "A readable backend error.");
    render(<LLMWorkspace />);
    await screen.findByText("Model ready");
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await screen.findByText("A readable backend error.");
    expect(screen.getByText(new RegExp(code))).toBeInTheDocument();
    expect(screen.getByText("Error details")).toBeInTheDocument();
    expect(posts()).toHaveLength(1);
  });
  it("handles actual fetch rejection without fallback science", async () => {
    fetchMock.mockRejectedValue(new TypeError("Network unavailable"));
    render(<LLMWorkspace />);
    await screen.findByText("API unavailable");
    expect(
      screen.queryByRole("button", { name: "Layer 12" }),
    ).not.toBeInTheDocument();
  });
  it("handles failed envelopes", async () => {
    current = {
      ...current,
      status: "failed",
      overview: null,
      error: {
        code: "TRACE_CAPTURE_FAILED",
        message: "Capture failed safely.",
      },
    };
    await runFailure();
    expect(screen.getByText("Capture failed safely.")).toBeInTheDocument();
    expect(posts()).toHaveLength(1);
  });
  it("expires visible traces without retracing", async () => {
    current.expires_at = new Date(Date.now() + 1400).toISOString();
    await run();
    await screen.findByText("Trace expired", {}, { timeout: 3000 });
    expect(
      screen.getByRole("button", { name: "Run trace again" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Layer 12" }),
    ).not.toBeInTheDocument();
    expect(posts()).toHaveLength(1);
  });
  it("turns detail 410 into the global expired state", async () => {
    await run();
    const normal = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((url, init) =>
      url.endsWith("/layers/12")
        ? Promise.resolve(err("TRACE_EXPIRED", "Expired by the backend", 410))
        : normal(url, init),
    );
    await userEvent.click(screen.getByRole("button", { name: "Layer 12" }));
    await screen.findByText("Trace expired");
    expect(posts()).toHaveLength(1);
  });
  it("handles deleted traces explicitly", async () => {
    await run();
    await userEvent.click(screen.getByText("Trace details"));
    await userEvent.click(screen.getByRole("button", { name: "Delete trace" }));
    await screen.findByText("Trace deleted");
    expect(posts()).toHaveLength(1);
  });
  it("supports arrow and end keyboard selection", async () => {
    await run();
    const node = screen.getByRole("button", { name: "Layer 12" });
    node.focus();
    fireEvent.keyDown(node, { key: "ArrowRight" });
    expect(screen.getByRole("button", { name: "Layer 13" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.keyDown(screen.getByRole("button", { name: "Layer 13" }), {
      key: "End",
    });
    await screen.findByText(/Real next-token candidates/);
  });
});
async function runFailure() {
  render(<LLMWorkspace />);
  await screen.findByText("Model ready");
  await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
  await screen.findByText("Capture failed safely.");
}

describe("trace isolation and lifecycle edges", () => {
  it("ignores a late manual status error for a replaced trace", async () => {
    current.created_at = new Date(Date.now() - 120001).toISOString();
    getOverview = () => ({ ...current, status: "running", overview: null });
    render(<LLMWorkspace />);
    await screen.findByText("Model ready");
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await screen.findByText("Trace is taking longer than expected");
    const previousId = current.trace_id;
    const normal = fetchMock.getMockImplementation()!;
    let finish: (response: Response) => void = () => {};
    fetchMock.mockImplementation((url, init) =>
      new URL(url).pathname === `/api/v1/traces/${previousId}`
        ? new Promise<Response>((resolve) => {
            finish = resolve;
          })
        : normal(url, init),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Check trace status" }),
    );
    current = {
      ...current,
      trace_id: "replacement-after-check",
      created_at: new Date().toISOString(),
    };
    getOverview = () => current;
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await screen.findByText("Trace ready");
    await act(async () =>
      finish(err("TRACE_EXPIRED", "The old trace expired.", 410)),
    );
    expect(screen.queryByText("Trace expired")).not.toBeInTheDocument();
    expect(screen.getByText("Trace ready")).toBeInTheDocument();
  });
  it("discards a late deletion result for an old trace", async () => {
    await run();
    const normal = fetchMock.getMockImplementation()!;
    let completeDelete: (response: Response) => void = () => {};
    fetchMock.mockImplementation((url, init) =>
      init?.method === "DELETE"
        ? new Promise<Response>((resolve) => {
            completeDelete = resolve;
          })
        : normal(url, init),
    );
    await userEvent.click(screen.getByText("Trace details"));
    await userEvent.click(screen.getByRole("button", { name: "Delete trace" }));
    current = { ...current, trace_id: "replacement" };
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await waitFor(() => expect(posts()).toHaveLength(2));
    await screen.findByText("Trace ready");
    await act(async () => completeDelete(new Response(null, { status: 204 })));
    expect(screen.queryByText("Trace deleted")).not.toBeInTheDocument();
    expect(screen.getByText("Trace ready")).toBeInTheDocument();
  });
  it("aborts an obsolete layer read and ignores its late response", async () => {
    await run();
    const normal = fetchMock.getMockImplementation()!;
    let aborted = false;
    let finish: (response: Response) => void = () => {};
    fetchMock.mockImplementation((url, init) =>
      url.endsWith("/layers/12")
        ? new Promise<Response>((resolve) => {
            finish = resolve;
            init!.signal!.addEventListener("abort", () => {
              aborted = true;
            });
          })
        : normal(url, init),
    );
    await userEvent.click(screen.getByRole("button", { name: "Layer 12" }));
    await screen.findByText("Loading Layer 12 from this trace…");
    current = { ...current, trace_id: "new-source" };
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await waitFor(() => expect(aborted).toBe(true));
    await act(async () => finish(json(layer)));
    await screen.findByText("Trace ready");
    expect(screen.getByRole("heading", { name: "Input" })).toBeInTheDocument();
    expect(screen.queryByText("Capture coverage")).not.toBeInTheDocument();
  });
  it("handles a component-not-captured response without rerunning", async () => {
    await run();
    const normal = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((url, init) =>
      url.endsWith("/components/embedding.output")
        ? Promise.resolve(
            err("COMPONENT_NOT_CAPTURED", "The raw component was omitted."),
          )
        : normal(url, init),
    );
    await userEvent.click(
      screen.getAllByRole("button", { name: "Embedding" })[0],
    );
    await screen.findByText("Not captured in this trace");
    expect(posts()).toHaveLength(1);
  });
  it("paginates real token metadata without truncating", async () => {
    const normal = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((url, init) =>
      new URL(url).pathname.endsWith("/tokens")
        ? Promise.resolve(
            json({
              ...tokens,
              items: new URL(url).searchParams.has("cursor")
                ? tokens.items.slice(2)
                : tokens.items.slice(0, 2),
              next_cursor: new URL(url).searchParams.has("cursor")
                ? null
                : "next-token-page",
            }),
          )
        : normal(url, init),
    );
    await run();
    expect(
      within(screen.getByRole("region", { name: "Input tokens" })).getAllByRole(
        "button",
      ),
    ).toHaveLength(5);
    expect(
      fetchMock.mock.calls.filter(([url]) =>
        new URL(url).pathname.endsWith("/tokens"),
      ),
    ).toHaveLength(2);
  });
  it("rejects whitespace-only prompts before submission", async () => {
    render(<LLMWorkspace />);
    await screen.findByText("Model ready");
    fireEvent.change(screen.getByLabelText("Raw prompt"), {
      target: { value: "   " },
    });
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await screen.findByText("Enter a prompt containing text.");
    expect(posts()).toHaveLength(0);
  });
  it("shows queued state and stops when ready", async () => {
    createResponse = () =>
      json({ ...current, status: "queued", overview: null }, 202);
    getOverview = () => ({ ...current, status: "queued", overview: null });
    render(<LLMWorkspace />);
    await screen.findByText("Model ready");
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await screen.findByText("Queued");
    getOverview = () => current;
    await screen.findByText("Trace ready");
    expect(posts()).toHaveLength(1);
  });
  it("bounds a long trace wait and permits a read-only status check", async () => {
    current.created_at = new Date(Date.now() - 120001).toISOString();
    getOverview = () => ({ ...current, status: "running", overview: null });
    render(<LLMWorkspace />);
    await screen.findByText("Model ready");
    await userEvent.click(screen.getByRole("button", { name: /TRACE MODEL/ }));
    await screen.findByText("Trace is taking longer than expected");
    expect(
      screen.getByRole("button", { name: "Cancel trace" }),
    ).toBeInTheDocument();
    getOverview = () => current;
    await userEvent.click(
      screen.getByRole("button", { name: "Check trace status" }),
    );
    await screen.findByText("Trace ready");
    expect(posts()).toHaveLength(1);
  });
});

describe("selection and cancellation", () => {
  it("resets all incompatible global identities atomically", () => {
    const store = createSelectionStore();
    store.getState().resetTrace("first");
    store.getState().selectToken(3);
    store.getState().selectLayer(12);
    store.getState().selectCandidate(10);
    store.setState({ selectedHead: 2, selectedGenerationStep: 1 });
    store.getState().resetTrace("second");
    expect(store.getState()).toMatchObject({
      traceId: "second",
      selectedToken: null,
      selectedLayer: null,
      selectedHead: null,
      selectedCandidateToken: null,
      selectedGenerationStep: null,
      selectedComponent: "input",
      activeVisualizationMode: "model-journey",
    });
  });
  it("passes external abort signals and preserves cancellation", async () => {
    const controller = new AbortController();
    const reason = new DOMException("Obsolete selection", "AbortError");
    fetchMock.mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init!.signal!.addEventListener("abort", () => reject(reason));
        }),
    );
    const pending = api.traceLayer("first", 12, controller.signal);
    controller.abort();
    await expect(pending).rejects.toBe(reason);
  });
});
