"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api/client";
import { overviewQuery, traceKey } from "@/lib/api/queries";
import { useSelection, useSelectionStore } from "@/state/selection";
import { WorkspaceProviders } from "./WorkspaceProviders";
import { TraceControls } from "@/components/trace/TraceControls";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { TraceDetails } from "@/components/trace/TraceDetails";
import { TokenStrip } from "@/components/tokens/TokenStrip";
import { ModelJourney } from "@/components/model-journey/ModelJourney";
import { Inspector } from "@/components/inspector/Inspector";

function Workspace() {
  const [prompt, setPrompt] = useState("The capital of Germany is");
  const [inputError, setInputError] = useState<Error | null>(null);
  const store = useSelectionStore();
  const id = useSelection((state) => state.traceId);
  const issue = useSelection((state) => state.traceIssue);
  const reset = useSelection((state) => state.resetTrace);
  const client = useQueryClient();
  const submission = useRef<AbortController | null>(null);
  useEffect(() => () => submission.current?.abort(), []);
  const model = useQuery({
    queryKey: ["model-status"],
    queryFn: ({ signal }) => api.modelStatus(signal),
    staleTime: 30000,
    refetchInterval: (query) =>
      query.state.data?.state === "loading" ? 1500 : false,
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
  });
  const load = useMutation({
    mutationFn: api.loadModel,
    onSuccess: (status) => {
      client.setQueryData(["model-status"], status);
    },
  });
  const trace = useQuery({ ...overviewQuery(id), enabled: !!id && !issue });
  const create = useMutation({
    mutationFn: (text: string) => {
      submission.current = new AbortController();
      return api.createTrace(
        { prompt: text, capture: { profile: "overview" }, display_top_k: 10 },
        false,
        submission.current.signal,
      );
    },
    onSuccess: (data) => {
      client.setQueryData(overviewQuery(data.trace_id).queryKey, data);
      reset(data.trace_id);
    },
  });
  const remove = useMutation({
    mutationFn: (traceId: string) => api.deleteTrace(traceId),
    onSuccess: (_response, deletedId) => {
      void client.cancelQueries({ queryKey: traceKey(deletedId) });
      client.removeQueries({ queryKey: traceKey(deletedId) });
      if (store.getState().traceId !== deletedId) return;
      store
        .getState()
        .setTraceIssue(
          new ApiError(
            "The retained computation has been released. Run an explicit new trace to continue.",
            "TRACE_DELETED",
          ),
        );
    },
  });
  const data = trace.data;
  const overview = data?.status === "ready" && !issue ? data.overview : null;
  const busy =
    create.isPending ||
    (!!id &&
      !issue &&
      (data?.status === "queued" ||
        data?.status === "running" ||
        trace.isPending));

  useEffect(() => {
    if (!data || issue) return;
    if (data.status === "ready" && data.expires_at) {
      const expire = () => {
        if (store.getState().traceId === data.trace_id) {
          store
            .getState()
            .setTraceIssue(
              new ApiError(
                "This trace reached its retention deadline. Run it again to inspect a new computation.",
                "TRACE_EXPIRED",
              ),
            );
          void client.cancelQueries({ queryKey: traceKey(data.trace_id) });
          client.removeQueries({ queryKey: traceKey(data.trace_id) });
        }
      };
      const delay = Math.max(
        0,
        new Date(data.expires_at).getTime() - Date.now(),
      );
      const timer = setTimeout(expire, Math.min(delay, 2147483647));
      const check = () => {
        if (Date.now() >= new Date(data.expires_at!).getTime()) expire();
      };
      window.addEventListener("focus", check);
      return () => {
        clearTimeout(timer);
        window.removeEventListener("focus", check);
      };
    }
    if (["queued", "running"].includes(data.status)) {
      const timer = setTimeout(
        () =>
          store
            .getState()
            .setTraceIssue(
              new ApiError(
                "The backend has not reported completion after two minutes. Check this trace's status or cancel it; no new execution will start automatically.",
                "TRACE_TIMEOUT",
              ),
            ),
        Math.max(0, new Date(data.created_at).getTime() + 120000 - Date.now()),
      );
      return () => clearTimeout(timer);
    }
  }, [data, issue, store, client]);

  function run() {
    if (busy || create.isPending) return;
    if (!prompt.trim()) {
      setInputError(
        new ApiError("Enter a prompt containing text.", "INVALID_REQUEST"),
      );
      return;
    }
    setInputError(null);
    create.reset();
    if (id) {
      void client.cancelQueries({ queryKey: traceKey(id) });
      client.removeQueries({ queryKey: traceKey(id) });
    }
    reset(null);
    create.mutate(prompt);
  }
  async function checkTrace() {
    const result = await trace.refetch();
    if (store.getState().traceId !== id) return;
    if (result.error instanceof ApiError)
      store.getState().setTraceIssue(result.error);
    else if (result.data && !["queued", "running"].includes(result.data.status))
      store.getState().setTraceIssue(null);
  }
  const failed =
    data?.status === "failed"
      ? new ApiError(
          typeof data.error?.message === "string"
            ? data.error.message
            : "The backend could not capture this trace.",
          typeof data.error?.code === "string"
            ? data.error.code
            : "TRACE_CAPTURE_FAILED",
          undefined,
          data.error ?? undefined,
        )
      : data?.status === "cancelled"
        ? new ApiError("The backend cancelled this trace.", "TRACE_DELETED")
        : null;
  const error = issue ?? inputError ?? create.error ?? trace.error ?? failed;
  const phase =
    issue?.code === "TRACE_EXPIRED"
      ? "Expired"
      : issue?.code === "TRACE_DELETED"
        ? "Deleted"
        : error
          ? "Error"
          : create.isPending
            ? "Submitting"
            : data?.status === "running"
              ? "Creating trace"
              : data?.status === "queued"
                ? "Queued"
                : overview
                  ? "Trace ready"
                  : "Awaiting trace";
  return (
    <main className="workspace">
      <TraceControls
        prompt={prompt}
        setPrompt={setPrompt}
        model={model.data}
        busy={busy}
        phase={phase}
        overview={overview}
        onRun={run}
        onLoad={() => load.mutate()}
        apiUnavailable={!!model.error}
      />
      {model.error && (
        <div className="workspace-notice">
          <ApiErrorNotice
            error={model.error}
            onRetry={() => void model.refetch()}
            retryLabel="Reconnect"
          />
        </div>
      )}
      {load.error && (
        <div className="workspace-notice">
          <ApiErrorNotice error={load.error} />
        </div>
      )}
      {model.data?.error && (
        <div className="workspace-notice">
          <ApiErrorNotice
            error={
              new ApiError(
                model.data.error,
                model.data.error_code ?? "MODEL_LOAD_FAILED",
              )
            }
          />
        </div>
      )}
      {error && (
        <div className="workspace-notice">
          <ApiErrorNotice
            error={error}
            onRetry={
              issue?.code === "TRACE_TIMEOUT"
                ? () => void checkTrace()
                : trace.error && !issue
                  ? () => void trace.refetch()
                  : run
            }
            retryLabel={
              issue?.code === "TRACE_TIMEOUT"
                ? "Check trace status"
                : trace.error && !issue
                  ? "Check trace"
                  : id || issue
                    ? "Run trace again"
                    : "Try again"
            }
          />
          {issue?.code === "TRACE_TIMEOUT" && id && (
            <button
              className="text-button"
              onClick={() => remove.mutate(id)}
              disabled={remove.isPending}
            >
              Cancel trace
            </button>
          )}
        </div>
      )}
      <section
        className="journey-region"
        aria-label="Model workspace"
        aria-busy={busy}
      >
        {overview && id && (
          <TokenStrip key={`tokens-${id}`} id={id} overview={overview} />
        )}
        <ModelJourney
          key={`journey-${id ?? "empty"}`}
          id={overview && id ? id : null}
          overview={overview}
          busy={busy}
        />
        {busy && (
          <div className="tracing-message" role="status">
            <span className="status-pulse" aria-hidden="true" />
            <div>
              <strong>Tracing Qwen3-0.6B…</strong>
              <p>Model inference and internal capture are running.</p>
            </div>
            {id && (
              <button
                className="text-button"
                onClick={() => remove.mutate(id)}
                disabled={remove.isPending}
              >
                Cancel trace
              </button>
            )}
          </div>
        )}
        {!overview && !busy && !error && (
          <p className="journey-invitation">
            Start with a prompt. Follow one computation from text to its
            next-token distribution.
          </p>
        )}
      </section>
      <Inspector
        key={`inspector-${overview && id ? id : "empty"}`}
        id={overview && id ? id : null}
        overview={overview}
      />
      {overview && data && id && (
        <TraceDetails
          trace={data}
          onDelete={() => remove.mutate(id)}
          deleting={remove.isPending}
          onVerify={() => void trace.refetch()}
        />
      )}
      {remove.error && (
        <div className="workspace-notice">
          <ApiErrorNotice error={remove.error} />
        </div>
      )}
      <footer className="workspace-footer">
        <span>One trace. A continuous model journey.</span>
        <span>Stage 7 · Residual Stream and MLP</span>
      </footer>
    </main>
  );
}
export function LLMWorkspace() {
  return (
    <WorkspaceProviders>
      <Workspace />
    </WorkspaceProviders>
  );
}
