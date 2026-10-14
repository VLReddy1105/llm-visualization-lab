import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { useEffect } from "react";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import type {
  AnalysisJob,
  AnalysisRequest,
  TraceEnvelope,
} from "@/lib/api/client";
import { Inspector } from "@/components/inspector/Inspector";
import { TokenStrip } from "@/components/tokens/TokenStrip";
import { WorkspaceProviders } from "@/components/workspace/WorkspaceProviders";
import { createSelectionStore, useSelectionStore } from "@/state/selection";

// Recorded real Qwen responses; faults below change lifecycle/transport only.
const evidence = JSON.parse(
  readFileSync(
    resolve(
      process.cwd(),
      "../../docs/validation/stage6-germany-analysis.json",
    ),
    "utf8",
  ),
) as { trace: TraceEnvelope; jobs: AnalysisJob[] };
const stage5 = JSON.parse(
  readFileSync(
    resolve(process.cwd(), "../../docs/validation/stage5-http-fixtures.json"),
    "utf8",
  ),
) as { method: string; path: string; response: unknown }[];
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
const error = (code: string, status = 409) =>
  json(
    { error: { code, message: `Transport test: ${code}`, retryable: false } },
    status,
  );
let source: TraceEnvelope;
let jobs: AnalysisJob[];
let store: ReturnType<typeof createSelectionStore>;
let fetchMock: ReturnType<typeof vi.fn>;
let fault:
  | ((url: URL, init?: RequestInit) => Response | Promise<Response> | undefined)
  | undefined;
let creations: number;

beforeEach(() => {
  source = structuredClone(evidence.trace);
  source.expires_at = new Date(Date.now() + 1800000).toISOString();
  jobs = structuredClone(evidence.jobs);
  for (const job of jobs) {
    job.created_at = new Date().toISOString();
    job.expires_at = new Date(Date.now() + 600000).toISOString();
  }
  fault = undefined;
  creations = 0;
  fetchMock = vi.fn(async (input: string, init?: RequestInit) => {
    const url = new URL(input),
      path = url.pathname;
    const failed = fault?.(url, init);
    if (failed) return failed;
    if (path === "/api/v1/traces" && init?.method === "POST") {
      creations++;
      return json({
        ...source,
        trace_id: "explicit-overview",
        overview: {
          ...source.overview,
          capture: { ...source.overview!.capture, profile: "overview" },
        },
      });
    }
    if (path.endsWith("/analyses") && init?.method === "POST") {
      const body: AnalysisRequest = JSON.parse(String(init.body));
      const job = jobs.find(
        (j) =>
          j.request.kind === body.kind &&
          j.request.parameters.token_index === body.parameters.token_index &&
          (body.kind !== "sampling_preview" ||
            (j.request.kind === "sampling_preview" &&
              j.request.parameters.temperature ===
                body.parameters.temperature)),
      );
      return job ? json(job, 202) : error("UNRECORDED_ANALYSIS");
    }
    if (/\/analyses\/[^/]+$/.test(path)) {
      const job = jobs.find((j) => path.endsWith(j.analysis_id));
      if (init?.method === "DELETE")
        return json({ ...job, status: "cancelled", result: null });
      return job ? json(job) : error("ANALYSIS_NOT_FOUND", 404);
    }
    if (path.endsWith("/tokens"))
      return json({
        items: source.overview!.input.tokens.items,
        total: 5,
        next_cursor: null,
      });
    if (/\/layers\/12$/.test(path))
      return json(stage5.find((r) => /\/layers\/12$/.test(r.path))!.response);
    if (path.endsWith("/explicit-overview"))
      return json({
        ...source,
        trace_id: "explicit-overview",
        overview: {
          ...source.overview,
          capture: { ...source.overview!.capture, profile: "overview" },
        },
      });
    return json(source);
  });
  vi.stubGlobal("fetch", fetchMock);
});
function Harness() {
  const selection = useSelectionStore();
  useEffect(() => {
    store = selection;
    selection.getState().resetTrace(source.trace_id);
    selection.getState().openEvolution("journey", 4);
  }, [selection]);
  return (
    <>
      <TokenStrip id={source.trace_id} overview={source.overview!} />
      <Inspector id={source.trace_id} overview={source.overview!} />
    </>
  );
}
const mount = () =>
  render(
    <WorkspaceProviders>
      <Harness />
    </WorkspaceProviders>,
  );
const view = (name: string) =>
  userEvent.click(
    within(
      screen.getByRole("navigation", { name: "Token evolution views" }),
    ).getByRole("button", { name }),
  );
const ready = () => screen.findByTestId("depth-analysis");

it("renders actual state order, preserves token/depth through microscope and reuses cached analyses", async () => {
  mount();
  await ready();
  const timeline = screen.getByRole("toolbar", {
    name: "Transformer depth checkpoints",
  });
  expect(within(timeline).getAllByRole("button")).toHaveLength(30);
  await userEvent.click(
    within(timeline).getByRole("button", {
      name: /POST-BLOCK\s*Layer 12/,
    }),
  );
  expect(store.getState().selectedLayer).toBe(12);
  fireEvent.keyDown(
    within(timeline).getByRole("button", {
      name: /POST-BLOCK\s*Layer 12/,
    }),
    { key: "ArrowRight" },
  );
  expect(store.getState().selectedLayer).toBe(13);
  fireEvent.keyDown(
    within(timeline).getByRole("button", {
      name: /POST-BLOCK\s*Layer 13/,
    }),
    { key: "ArrowLeft" },
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Open Layer 12 in Microscope" }),
  );
  await screen.findByText(
    "Two residual additions. One continuous representation.",
  );
  await view("Representation Evolution");
  await ready();
  expect(store.getState().selectedToken).toBe(4);
  expect(store.getState().selectedDepth).toBe(13);
  const posts = fetchMock.mock.calls.filter(
    ([u, i]) => String(u).endsWith("/analyses") && i?.method === "POST",
  );
  expect(posts).toHaveLength(1);
  expect(creations).toBe(0);
});
it("shows actual lens provenance, candidates, rank and probabilities", async () => {
  mount();
  await ready();
  await view("Candidate Evolution");
  await screen.findByText("Final projection agrees with retained logits");
  expect(
    screen.getByRole("region", {
      name: "Candidate Probability Evolution",
    }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("region", {
      name: "Candidate Logit Evolution",
    }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("region", {
      name: "Candidate Rank Evolution",
    }),
  ).toBeInTheDocument();
  expect(
    screen.getByText(
      /These diagnostic projections are not actual intermediate predictions/,
    ),
  ).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: 'Token 0: "The"' }));
  await screen.findByText(
    "Original logits comparison unavailable at this position.",
  );
  expect(store.getState().selectedComponent).toBe("evolution");
  expect(creations).toBe(0);
});
it("keeps sampling settings on return and resets without another trace", async () => {
  mount();
  await ready();
  await view("Sampling Lab");
  await screen.findByText(/Applied: temperature 1/);
  fireEvent.change(screen.getByLabelText("Temperature"), {
    target: { value: "0.7" },
  });
  fireEvent.change(screen.getByLabelText("Top-k (0 disables)"), {
    target: { value: "20" },
  });
  fireEvent.change(screen.getByLabelText("Top-p"), {
    target: { value: "0.8" },
  });
  await userEvent.click(screen.getByRole("button", { name: "Apply preview" }));
  await screen.findByText(/Applied: temperature 0.7/);
  await view("Token Journey");
  await ready();
  await view("Sampling Lab");
  expect(screen.getByLabelText("Temperature")).toHaveValue(0.7);
  await userEvent.click(screen.getByRole("button", { name: "Reset defaults" }));
  await screen.findByText(/Applied: temperature 1/);
  expect(creations).toBe(0);
});
it.each([
  "ANALYSIS_EXPIRED",
  "ANALYSIS_NOT_FOUND",
  "QUEUE_FULL",
  "MODEL_TRACE_MISMATCH",
  "COMPONENT_NOT_CAPTURED",
])("shows %s without scientific fallback", async (code) => {
  fault = (url) =>
    url.pathname.endsWith("/analyses") ? error(code) : undefined;
  mount();
  await screen.findByText(`Transport test: ${code}`);
  expect(screen.queryByTestId("depth-analysis")).not.toBeInTheDocument();
  expect(creations).toBe(0);
});
it("hides a ready analysis at its absolute expiry", async () => {
  jobs[0].expires_at = new Date(Date.now() + 600).toISOString();
  mount();
  await ready();
  await screen.findByText(/This analysis expired/, {}, { timeout: 3000 });
  expect(screen.queryByTestId("depth-analysis")).not.toBeInTheDocument();
});
it("cancels a running analysis and hides partial science", async () => {
  jobs[0].status = "running";
  jobs[0].result = null;
  mount();
  await screen.findByText(/Analyzing saved observations/);
  await userEvent.click(
    await screen.findByRole("button", { name: "Cancel analysis" }),
  );
  await screen.findByText("Analysis was cancelled.");
  expect(screen.queryByTestId("depth-analysis")).not.toBeInTheDocument();
});
it("requires explicit recapture and keeps the new overview identity in the microscope", async () => {
  source.overview!.capture.profile = "summary";
  mount();
  await screen.findByText(/Complete depth vectors were not captured/);
  expect(creations).toBe(0);
  await userEvent.click(
    screen.getByRole("button", { name: "Capture Token Evolution Overview" }),
  );
  await ready();
  expect(creations).toBe(1);
  expect(
    store.getState().evolutionTraces[`${source.trace_id}:-1:overview`],
  ).toBe("explicit-overview");
  await userEvent.click(
    within(screen.getByRole("toolbar")).getByRole("button", {
      name: /POST-BLOCK\s*Layer 12/,
    }),
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Open Layer 12 in Microscope" }),
  );
  await screen.findByText("Source trace: explicit-overview");
});
it("does not attach late recapture acknowledgements to a replaced source", async () => {
  source.overview!.capture.profile = "summary";
  let release: (response: Response) => void = () => {};
  fault = (url, init) =>
    url.pathname === "/api/v1/traces" && init?.method === "POST"
      ? new Promise<Response>((resolve) => {
          release = resolve;
        })
      : undefined;
  mount();
  await userEvent.click(
    await screen.findByRole("button", {
      name: "Capture Token Evolution Overview",
    }),
  );
  act(() => store.getState().resetTrace("unrelated-source"));
  await act(async () =>
    release(json({ ...source, trace_id: "late-overview" })),
  );
  expect(store.getState().evolutionTraces).toEqual({});
});
it("resets all trace-scoped depth, candidate and sampling state", () => {
  const s = createSelectionStore();
  s.getState().resetTrace("a");
  s.getState().selectToken(3);
  s.getState().selectLayer(12, "detail");
  s.getState().pinCandidate(220);
  s.getState().associateEvolution("a:-1:overview", "detail");
  s.getState().resetTrace("b");
  expect(s.getState().selectedDepth).toBeNull();
  expect(s.getState().microscopeTraceId).toBeNull();
  expect(s.getState().pinnedCandidates).toEqual([]);
  expect(s.getState().evolutionTraces).toEqual({});
});

it("recovers from an analysis transport error without capturing another trace", async () => {
  fault = (url) =>
    url.pathname.endsWith("/analyses") ? error("QUEUE_FULL", 429) : undefined;
  mount();
  await screen.findByText("Transport test: QUEUE_FULL");
  fault = undefined;
  await userEvent.click(
    screen.getByRole("button", { name: "Recompute saved-data analysis" }),
  );
  await ready();
  expect(creations).toBe(0);
});
it("stops polling after the analysis deadline and offers a read-only status check", async () => {
  jobs[0].status = "running";
  jobs[0].result = null;
  jobs[0].created_at = new Date(Date.now() - 121000).toISOString();
  mount();
  await screen.findByText("Analysis has not completed within two minutes.", {
    exact: false,
  });
  expect(
    screen.getByRole("button", { name: "Check analysis status" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Cancel analysis" }),
  ).toBeInTheDocument();
  expect(creations).toBe(0);
});
it("reports an invalid pinned vocabulary ID without requesting a new analysis", async () => {
  mount();
  await ready();
  await view("Logit Lens");
  await screen.findByText("Final projection agrees with retained logits");
  const before = fetchMock.mock.calls.length;
  fireEvent.change(screen.getByLabelText("Pin vocabulary ID"), {
    target: { value: "99999999" },
  });
  await userEvent.click(
    screen.getByRole("button", { name: "Toggle pinned ID" }),
  );
  await screen.findByText(
    "Enter a valid vocabulary ID; at most eight pinned candidates.",
  );
  expect(fetchMock.mock.calls.length).toBe(before);
});
