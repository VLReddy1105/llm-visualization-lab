import { useEffect } from "react";
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
import { api, type TraceEnvelope } from "@/lib/api/client";
import { WorkspaceProviders } from "@/components/workspace/WorkspaceProviders";
import { Inspector } from "@/components/inspector/Inspector";
import { TokenStrip } from "@/components/tokens/TokenStrip";
import { createSelectionStore, useSelectionStore } from "@/state/selection";

// HTTP playback only: every numerical observation below was recorded from real Qwen.
type Recording = {
  method: string;
  path: string;
  body: unknown;
  response: unknown;
};
const evidence = (name: string) =>
  JSON.parse(
    readFileSync(resolve(process.cwd(), "../../docs/validation", name), "utf8"),
  );
const records: Recording[] = evidence("stage5-http-fixtures.json");
const example: {
  overviewId: string;
  deepId: string;
  relation: { observations: { weights_reconstructed: { values: number[] } } };
  norms: { q: string; k: string; v: string };
} = evidence("stage5-germany-browser.json");
const sourceRecord = records.find(
  (r) =>
    r.method === "GET" &&
    r.path.startsWith(`/api/v1/traces/${example.overviewId}?`) &&
    (r.response as TraceEnvelope).status === "ready",
)!.response as TraceEnvelope;
const deepRecord = records.find(
  (r) =>
    r.method === "GET" &&
    r.path.startsWith(`/api/v1/traces/${example.deepId}?`) &&
    (r.response as TraceEnvelope).status === "ready",
)!.response as TraceEnvelope;
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
const error = (code: string, status = 409) =>
  json(
    {
      error: {
        code,
        message: `Test transport failure: ${code}`,
        retryable: false,
      },
    },
    status,
  );
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object")
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k}:${canonical(v)}`)
      .join(",")}}`;
  return String(value);
}
let source: TraceEnvelope;
let deep: TraceEnvelope;
let store: ReturnType<typeof createSelectionStore>;
let interceptor:
  ((path: string, init?: RequestInit) => Response | undefined) | undefined;
let fetchMock: ReturnType<typeof vi.fn>;
let creations: number;
let unresolved: string[];

beforeEach(() => {
  source = structuredClone(sourceRecord);
  deep = structuredClone(deepRecord);
  for (const envelope of [source, deep]) {
    envelope.created_at = new Date().toISOString();
    envelope.expires_at = new Date(Date.now() + 1800000).toISOString();
  }
  creations = 0;
  unresolved = [];
  interceptor = undefined;
  fetchMock = vi.fn(async (input: string, init?: RequestInit) => {
    const url = new URL(input);
    const path = decodeURIComponent(url.pathname + url.search);
    const intercepted = interceptor?.(path, init);
    if (intercepted) return intercepted;
    const method = init?.method ?? "GET";
    if (url.pathname === "/api/v1/traces" && method === "POST") {
      creations++;
      return json({ ...deep, status: "running", overview: null }, 202);
    }
    if (
      url.pathname === `/api/v1/traces/${deep.trace_id}` &&
      method === "DELETE"
    )
      return new Response(null, { status: 204 });
    if (url.pathname === `/api/v1/traces/${deep.trace_id}`) return json(deep);
    if (url.pathname === `/api/v1/traces/${source.trace_id}`)
      return json(source);
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    const item = [...records]
      .reverse()
      .find(
        (r) =>
          r.method === method &&
          decodeURIComponent(r.path) === path &&
          canonical(r.body) === canonical(body),
      );
    if (item) return json(item.response);
    unresolved.push(`${method} ${path} ${canonical(body)}`);
    return error("UNRECORDED_TEST_REQUEST", 500);
  });
  vi.stubGlobal("fetch", fetchMock);
});

function Harness({ alreadyCaptured = false }: { alreadyCaptured?: boolean }) {
  const selection = useSelectionStore();
  useEffect(() => {
    store = selection;
    const state = selection.getState();
    state.resetTrace(source.trace_id);
    state.selectToken(4);
    state.selectLayer(12);
    state.selectHead(4);
    state.selectAttention(12, 4);
    state.selectRelation(4, 3);
    if (alreadyCaptured)
      state.associateAttention(
        `${source.trace_id}:12:attention`,
        deep.trace_id,
      );
  }, [selection, alreadyCaptured]);
  return (
    <>
      <button onClick={() => selection.getState().selectLayer(13)}>
        Open Layer 13
      </button>
      <TokenStrip id={source.trace_id} overview={source.overview!} />
      <Inspector id={source.trace_id} overview={source.overview!} />
    </>
  );
}
const mount = (alreadyCaptured = false) =>
  render(
    <WorkspaceProviders>
      <Harness alreadyCaptured={alreadyCaptured} />
    </WorkspaceProviders>,
  );
async function capture() {
  await userEvent.click(
    await screen.findByRole("button", { name: "Capture Layer 12 Attention" }),
  );
  await screen.findByRole("grid");
  await screen.findByTestId("relation-weight");
}

describe("Stage 5 attention exploration", () => {
  it("requires explicit separate execution with the exact raw prompt and attention plan", async () => {
    mount();
    await screen.findByText(
      "Detailed attention data was not captured in this trace.",
    );
    expect(creations).toBe(0);
    await capture();
    expect(creations).toBe(1);
    const call = fetchMock.mock.calls.find(
      ([url, init]) =>
        new URL(url).pathname === "/api/v1/traces" && init?.method === "POST",
    )!;
    expect(JSON.parse(String(call[1].body))).toEqual({
      prompt: source.overview!.input.text,
      capture: { profile: "attention", layers: [12] },
      display_top_k: 10,
    });
    expect(store.getState().traceId).toBe(source.trace_id);
    expect(
      store.getState().attentionTraces[`${source.trace_id}:12:attention`],
    ).toBe(deep.trace_id);
    expect(
      screen.getByText("ATTENTION CAPTURE · SEPARATE EXECUTION"),
    ).toBeVisible();
    expect(unresolved).toEqual([]);
  });
  it("loads an existing detail association without a model creation", async () => {
    mount(true);
    await screen.findByRole("grid");
    expect(creations).toBe(0);
    expect(
      screen.queryByRole("button", { name: "Capture Layer 12 Attention" }),
    ).not.toBeInTheDocument();
  });
  it("renders 16 query heads and the backend GQA mapping, switching 4 → 7 → 4", async () => {
    mount(true);
    await screen.findByRole("grid");
    expect(
      within(
        screen.getByRole("group", { name: "Query head selection" }),
      ).getAllByRole("button"),
    ).toHaveLength(16);
    expect(await screen.findByText("Query Head 4 → KV Head 2")).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Query Head 7" }));
    expect(await screen.findByText("Query Head 7 → KV Head 3")).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Query Head 4" }));
    expect(await screen.findByText("Query Head 4 → KV Head 2")).toBeVisible();
    expect(creations).toBe(0);
    expect(unresolved).toEqual([]);
  });
  it("shows reconstructed cells identical to the retained mathematical relation", async () => {
    mount(true);
    const grid = await screen.findByRole("grid");
    const cell = grid.querySelector<HTMLButtonElement>(
      '[data-query="4"][data-key="3"]',
    )!;
    const weight = String(
      example.relation.observations.weights_reconstructed.values[0],
    );
    expect(cell.dataset.weight).toBe(weight);
    expect(await screen.findByTestId("relation-weight")).toHaveTextContent(
      weight,
    );
    expect(cell).toHaveAttribute(
      "aria-label",
      expect.stringContaining('native "ĠGermany"; token ID 9856'),
    );
    expect(screen.getByTestId("qk-score")).toHaveTextContent(
      "78.73936462402344",
    );
    expect(screen.getByTestId("scaled-score")).toHaveTextContent(
      "6.95964241027832",
    );
  });
  it("keeps native masked cells distinct and rejects click selection", async () => {
    mount(true);
    const grid = await screen.findByRole("grid");
    expect(grid.querySelectorAll('[data-masked="true"]')).toHaveLength(10);
    fireEvent.click(grid.querySelector('[data-query="3"][data-key="4"]')!);
    expect(store.getState().selectedQueryToken).toBe(4);
    expect(store.getState().selectedKeyToken).toBe(3);
    const cell = grid.querySelector<HTMLButtonElement>(
      '[data-query="4"][data-key="3"]',
    )!;
    cell.focus();
    fireEvent.keyDown(cell, { key: "ArrowLeft" });
    expect(grid.querySelector('[data-query="4"][data-key="2"]')).toHaveFocus();
  });
  it("synchronizes input/query selection and removes future connections", async () => {
    mount(true);
    await screen.findByRole("grid");
    await userEvent.click(
      screen.getByRole("button", { name: 'Token 3: " Germany"' }),
    );
    expect(store.getState().selectedComponent).toBe("attention");
    expect(screen.getByLabelText("Query token", { exact: true })).toHaveValue(
      "3",
    );
    await waitFor(() =>
      expect(document.querySelectorAll(".connection-keys button")).toHaveLength(
        4,
      ),
    );
    await userEvent.selectOptions(
      screen.getByLabelText("Key token", { exact: true }),
      "4",
    );
    expect(await screen.findByText(/Causally masked · this key/)).toBeVisible();
    expect(screen.queryByTestId("relation-weight")).not.toBeInTheDocument();
    expect(unresolved).toEqual([]);
  });
  it("loads bounded Q/K/V and head-output vectors with selected statistics and raw disclosure", async () => {
    mount(true);
    await screen.findByTestId("relation-weight");
    await userEvent.click(
      screen.getByRole("button", { name: "Inspect Q / K / V vectors" }),
    );
    expect(await screen.findByTestId("q-l2_norm")).toHaveTextContent(
      example.norms.q,
    );
    expect(await screen.findByTestId("k-l2_norm")).toHaveTextContent(
      example.norms.k,
    );
    expect(await screen.findByTestId("v-l2_norm")).toHaveTextContent(
      example.norms.v,
    );
    await userEvent.click(
      screen.getByText("Show raw values · Q after RoPE", { exact: true }),
    );
    const region = screen.getByRole("region", { name: "Q after RoPE vector" });
    expect(within(region).getByText(/"values": \[/)).toBeVisible();
    for (const stage of ["projected", "normalized", "rotary"]) {
      await userEvent.selectOptions(
        screen.getByLabelText("Q / K observation stage"),
        stage,
      );
      expect(await screen.findByTestId("q-l2_norm")).toBeVisible();
    }
    expect(unresolved).toEqual([]);
  });
  it("reuses captured Layer 12 on return but requires explicit Layer 13 capture", async () => {
    mount();
    await capture();
    await userEvent.click(
      screen.getByRole("button", { name: "← Back to Layer" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: /02Attention/ }),
    );
    await screen.findByRole("grid");
    expect(creations).toBe(1);
    expect(store.getState().selectedHead).toBe(4);
    expect(store.getState().selectedKeyToken).toBe(3);
    await userEvent.click(
      screen.getByRole("button", { name: "Open Layer 13" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: /02Attention/ }),
    );
    expect(
      await screen.findByRole("button", { name: "Capture Layer 13 Attention" }),
    ).toBeVisible();
    expect(creations).toBe(1);
    await userEvent.click(
      screen.getByRole("button", { name: "Back to Model Journey" }),
    );
    expect(store.getState().activeVisualizationMode).toBe("model-journey");
  });
  it("removes expired data and requires explicit recapture", async () => {
    deep.expires_at = new Date(Date.now() + 600).toISOString();
    mount(true);
    await screen.findByRole("grid");
    expect(
      await screen.findByText("Detailed trace expired", {}, { timeout: 2000 }),
    ).toBeVisible();
    expect(screen.queryByRole("grid")).not.toBeInTheDocument();
    expect(
      await screen.findByRole("button", {
        name: "Capture Layer 12 Attention again",
      }),
    ).toBeVisible();
    expect(creations).toBe(0);
    expect(store.getState().traceIssue).toBeNull();
  });
  it.each(["TRACE_EXPIRED", "TRACE_EVICTED", "TRACE_NOT_FOUND"])(
    "handles %s without stale tensors or harming the overview",
    async (code) => {
      interceptor = (path) =>
        path.endsWith("/heads/7")
          ? error(code, code === "TRACE_NOT_FOUND" ? 404 : 410)
          : undefined;
      mount(true);
      await screen.findByRole("grid");
      await userEvent.click(
        screen.getByRole("button", { name: "Query Head 7" }),
      );
      await screen.findByRole("button", {
        name: "Capture Layer 12 Attention again",
      });
      expect(screen.queryByRole("grid")).not.toBeInTheDocument();
      expect(store.getState().traceIssue).toBeNull();
      expect(creations).toBe(0);
    },
  );
  it("shows missing components explicitly without substituting values", async () => {
    interceptor = (path) =>
      path.includes("weights_reconstructed/slice")
        ? error("COMPONENT_NOT_CAPTURED")
        : undefined;
    mount(true);
    await screen.findAllByText(
      "Test transport failure: COMPONENT_NOT_CAPTURED",
    );
    expect(screen.queryByRole("grid")).not.toBeInTheDocument();
    expect(creations).toBe(0);
  });
  it("surfaces output mismatch and comparison failure", async () => {
    const comparison = records.find((r) => r.path.includes("/compare-output/"))!
      .response as Record<string, unknown>;
    interceptor = (path) =>
      path.includes("/compare-output/")
        ? json({ ...comparison, passed: false, same_top_token: false })
        : undefined;
    mount(true);
    expect(
      await screen.findByText(/Warning: these executions do not match/),
    ).toBeVisible();
    expect(await screen.findByRole("grid")).toBeVisible();
  });
  it("shows comparison unavailability without assuming outputs match", async () => {
    interceptor = (path) =>
      path.includes("/compare-output/")
        ? error("TRACE_EXPIRED", 410)
        : undefined;
    mount(true);
    expect(
      await screen.findByText(/Output comparison unavailable/),
    ).toBeVisible();
    expect(await screen.findByRole("grid")).toBeVisible();
    expect(store.getState().traceIssue).toBeNull();
  });
  it("reports capture failures and does not resubmit automatically", async () => {
    deep.status = "failed";
    deep.overview = null;
    deep.error = { code: "CAPTURE_FAILED" };
    mount();
    await userEvent.click(
      await screen.findByRole("button", { name: "Capture Layer 12 Attention" }),
    );
    expect(await screen.findByText(/Detailed capture failed/)).toBeVisible();
    expect(screen.queryByRole("grid")).not.toBeInTheDocument();
    expect(creations).toBe(1);
  });
  it("supports pending capture cancellation without creating another execution", async () => {
    deep.status = "running";
    deep.overview = null;
    mount(true);
    await userEvent.click(
      await screen.findByRole("button", { name: "Cancel detailed capture" }),
    );
    expect(await screen.findByText("Detailed trace unavailable")).toBeVisible();
    expect(creations).toBe(0);
  });
  it("resets all associations and attention selections for a new overview", async () => {
    mount(true);
    await screen.findByRole("grid");
    act(() => store.getState().resetTrace("new-overview"));
    const state = store.getState();
    expect(state.attentionTraces).toEqual({});
    expect(state.selectedHead).toBeNull();
    expect(state.selectedQueryToken).toBeNull();
    expect(state.selectedKeyToken).toBeNull();
  });
  it("stops a long-running capture and allows a read-only status check", async () => {
    deep.status = "running";
    deep.overview = null;
    deep.created_at = new Date(Date.now() - 121000).toISOString();
    mount(true);
    expect(
      await screen.findByText(/Capture is taking longer than two minutes/),
    ).toBeVisible();
    deep.status = "ready";
    deep.overview = structuredClone(deepRecord.overview);
    await userEvent.click(
      screen.getByRole("button", { name: "Check detailed trace" }),
    );
    await screen.findByRole("grid");
    expect(creations).toBe(0);
  });
  it("shows budget errors without retrying a capture automatically", async () => {
    interceptor = (path, init) =>
      path.startsWith("/api/v1/traces?") && init?.method === "POST"
        ? error("TRACE_BUDGET_EXCEEDED", 413)
        : undefined;
    mount();
    await userEvent.click(
      await screen.findByRole("button", { name: "Capture Layer 12 Attention" }),
    );
    expect(
      await screen.findByText("Test transport failure: TRACE_BUDGET_EXCEEDED"),
    ).toBeVisible();
    expect(screen.queryByRole("grid")).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter(
        ([url, init]) =>
          new URL(url).pathname === "/api/v1/traces" && init?.method === "POST",
      ),
    ).toHaveLength(1);
  });
  it("ignores a late capture acknowledgement after the source overview changes", async () => {
    let complete!: (value: TraceEnvelope) => void;
    vi.spyOn(api, "createTrace").mockImplementation(
      () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    );
    mount();
    await userEvent.click(
      await screen.findByRole("button", { name: "Capture Layer 12 Attention" }),
    );
    act(() => store.getState().resetTrace("replacement-overview"));
    await act(async () => {
      complete(deep);
    });
    expect(store.getState().attentionTraces).toEqual({});
    expect(store.getState().traceId).toBe("replacement-overview");
  });
});
