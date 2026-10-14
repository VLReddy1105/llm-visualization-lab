import { useEffect, useRef } from "react";
import type { Checkpoint, TraceOverview } from "@/lib/api/client";
import { ApiError } from "@/lib/api/client";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { useAttentionCapture } from "@/components/attention/useAttentionCapture";
import { useSelection, type EvolutionView } from "@/state/selection";
import { useAnalysis } from "./useAnalysis";
import { AnalysisStatus } from "./AnalysisStatus";
import { RepresentationView } from "./RepresentationView";
import { LensView } from "./LensView";
import { SamplingLab } from "./SamplingLab";

export const evolutionViews: [EvolutionView, string][] = [
  ["journey", "Token Journey"],
  ["representation", "Representation Evolution"],
  ["lens", "Logit Lens"],
  ["candidates", "Candidate Evolution"],
  ["sampling", "Sampling Lab"],
];

export function EvolutionNavigation({
  fallbackToken,
}: {
  fallbackToken: number;
}) {
  const open = useSelection((s) => s.openEvolution);
  const active = useSelection((s) =>
    s.selectedComponent === "evolution" ? s.evolutionView : null,
  );
  return (
    <nav className="evolution-navigation" aria-label="Token evolution views">
      {evolutionViews.map(([view, label]) => (
        <button
          className="secondary-button"
          key={view}
          aria-pressed={active === view}
          onClick={() => open(view, fallbackToken)}
        >
          {label}
        </button>
      ))}
    </nav>
  );
}

function DepthTimeline({
  checkpoints,
  selected,
  onSelect,
}: {
  checkpoints: Checkpoint[];
  selected: number;
  onSelect: (i: number) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    container.current
      ?.querySelector<HTMLButtonElement>(`[data-depth="${selected}"]`)
      ?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [selected]);
  return (
    <div
      className="depth-timeline"
      ref={container}
      role="toolbar"
      aria-label="Transformer depth checkpoints"
    >
      {checkpoints.map((c, i) => (
        <button
          key={c.component_id}
          className="depth-stop"
          data-depth={i}
          tabIndex={i === selected ? 0 : -1}
          aria-pressed={i === selected}
          title={c.component_id}
          onClick={() => onSelect(i)}
          onKeyDown={(e) => {
            const next =
              e.key === "Home"
                ? 0
                : e.key === "End"
                  ? checkpoints.length - 1
                  : e.key === "ArrowRight"
                    ? Math.min(i + 1, checkpoints.length - 1)
                    : e.key === "ArrowLeft"
                      ? Math.max(i - 1, 0)
                      : null;
            if (next != null) {
              e.preventDefault();
              onSelect(next);
              container.current
                ?.querySelector<HTMLButtonElement>(`[data-depth="${next}"]`)
                ?.focus();
            }
          }}
        >
          <small>
            {c.stage === "embedding"
              ? "INPUT"
              : c.stage === "final_normalized"
                ? "READOUT"
                : "POST-BLOCK"}
          </small>
          {c.label}
        </button>
      ))}
    </div>
  );
}

function DepthAnalysis({
  id,
  token,
  view,
}: {
  id: string;
  token: number;
  view: EvolutionView;
}) {
  const pins = useSelection((s) => s.pinnedCandidates);
  const depth = useSelection((s) => s.selectedDepth);
  const selectDepth = useSelection((s) => s.selectDepth);
  const openLayer = useSelection((s) => s.selectLayer);
  const isLens = view === "lens" || view === "candidates";
  const analysis = useAnalysis(
    id,
    isLens
      ? {
          kind: "logit_lens",
          parameters: {
            token_index: token,
            candidate_ids: pins,
            display_top_k: 5,
            recipe: "final_norm_then_lm_head",
          },
        }
      : { kind: "token_evolution", parameters: { token_index: token } },
  );
  const result = analysis.ready ? analysis.data?.result : null;
  useEffect(() => {
    if (result && (result.kind === "token_evolution" || result.kind === "logit_lens") && depth == null) {
      const index = result.checkpoints.length - 2;
      selectDepth(index, result.checkpoints[index].layer_index);
    }
  }, [result, depth, selectDepth]);
  if (!result || (result.kind !== "token_evolution" && result.kind !== "logit_lens"))
    return <AnalysisStatus analysis={analysis} />;
  const selected =
    depth != null && depth < result.checkpoints.length
      ? depth
      : result.checkpoints.length - 2;
  const active = result.checkpoints[selected];
  const select = (i: number) =>
    selectDepth(i, result.checkpoints[i].layer_index);
  return (
    <div
      className="depth-analysis"
      data-testid="depth-analysis"
      data-token-index={result.source.token.token_index}
    >
      <div className="evolution-context">
        <div>
          <p className="eyebrow">
            SELECTED INPUT POSITION {result.source.token.token_index}
          </p>
          <h4>
            {JSON.stringify(result.source.token.decoded_token)}{" "}
            <small>
              #{result.source.token.token_id} ·{" "}
              {result.source.token.tokenizer_piece}
            </small>
          </h4>
        </div>
        <span className="evidence-label">
          DERIVED · {result.source.device} / {result.source.dtype}
        </span>
      </div>
      <DepthTimeline
        checkpoints={result.checkpoints}
        selected={selected}
        onSelect={select}
      />
      <div className="depth-actions">
        <span>
          {active.label} · {result.source.layer_count} transformer blocks
        </span>
        {active.layer_index != null && (
          <button
            className="text-button"
            onClick={() => openLayer(active.layer_index!, id)}
          >
            Open Layer {active.layer_index} in Microscope
          </button>
        )}
      </div>
      {result.kind === "token_evolution" ? (
        <RepresentationView
          data={result}
          selected={selected}
          onSelect={select}
          expanded={view === "representation"}
        />
      ) : (
        <LensView
          data={result}
          selected={selected}
          onSelect={select}
          trajectories={view === "candidates"}
        />
      )}
      <details className="analysis-provenance">
        <summary>Analysis identity and model configuration</summary>
        <dl className="facts">
          <div>
            <dt>Source trace</dt>
            <dd>{result.source.trace_id}</dd>
          </div>
          <div>
            <dt>Analysis</dt>
            <dd>
              {analysis.data?.analysis_id} · {analysis.data?.method_version}
            </dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>{result.source.hf_model_id}</dd>
          </div>
          <div>
            <dt>Revision</dt>
            <dd>{result.source.revision}</dd>
          </div>
          <div>
            <dt>Shape</dt>
            <dd>
              {result.source.layer_count} layers · {result.source.hidden_size}{" "}
              hidden coordinates · {result.source.vocabulary_size} vocabulary
              entries
            </dd>
          </div>
          <div>
            <dt>Configuration digest</dt>
            <dd>{result.source.config_digest}</dd>
          </div>
          <div>
            <dt>Expires</dt>
            <dd>{analysis.data?.expires_at}</dd>
          </div>
        </dl>
      </details>
    </div>
  );
}

export function EvolutionWorkspace({
  sourceId,
  source,
}: {
  sourceId: string;
  source: TraceOverview;
}) {
  const token =
    useSelection((s) => s.selectedToken) ?? source.input.sequence_length - 1;
  const view = useSelection((s) => s.evolutionView);
  const capture = useAttentionCapture(sourceId, -1, source, "overview");
  if (view === "sampling")
    return <SamplingLab key={sourceId} id={sourceId} source={source} />;
  const failure =
    capture.issue ??
    capture.capture.error ??
    capture.trace.error ??
    (capture.trace.data?.status === "failed"
      ? new ApiError(
          String(
            capture.trace.data.error?.message ?? "Overview capture failed.",
          ),
          "TRACE_CAPTURE_FAILED",
        )
      : null);
  if (failure || !capture.ready)
    return (
      <div className="evolution-missing">
        {failure && <ApiErrorNotice error={failure} />}
        {capture.pending ? (
          <>
            <p role="status">Capturing overview vectors…</p>
            <button
              className="text-button"
              onClick={() => capture.cancel.mutate()}
            >
              Cancel capture
            </button>
            {capture.timedOut && (
              <button
                className="secondary-button"
                onClick={() => void capture.trace.refetch()}
              >
                Check capture status
              </button>
            )}
          </>
        ) : (
          <>
            <p>
              Complete depth vectors were not captured in this trace. An
              explicit overview capture will create a separate execution of the
              same raw prompt.
            </p>
            <button
              className="primary-button"
              onClick={() => capture.capture.mutate()}
              disabled={capture.capture.isPending}
            >
              Capture Token Evolution Overview
            </button>
          </>
        )}
      </div>
    );
  return (
    <>
      {capture.id !== sourceId && (
        <p className="capture-lineage">
          Separate overview execution: {capture.id}. Original trace: {sourceId}.
          Metrics and projections below use the new overview only.
        </p>
      )}
      <DepthAnalysis
        key={`${capture.id}-${token}-${view === "lens" || view === "candidates" ? "lens" : "metrics"}`}
        id={capture.id!}
        token={token}
        view={view}
      />
    </>
  );
}
