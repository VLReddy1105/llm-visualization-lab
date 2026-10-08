import type { TraceEnvelope } from "@/lib/api/client";
import { number, RawData } from "@/components/ui/ScientificData";

export function TraceDetails({
  trace,
  onDelete,
  deleting,
  onVerify,
}: {
  trace: TraceEnvelope;
  onDelete: () => void;
  deleting: boolean;
  onVerify: () => void;
}) {
  const overview = trace.overview;
  if (!overview) return null;
  return (
    <details className="trace-details">
      <summary>
        Trace details{" "}
        <span>
          {overview.model.device} · {number(overview.summary.duration_ms)} ms ·{" "}
          {overview.capture.profile}
        </span>
      </summary>
      <dl className="facts">
        <div>
          <dt>Trace ID</dt>
          <dd>{trace.trace_id}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{overview.model.hf_model_id}</dd>
        </div>
        <div>
          <dt>Revision</dt>
          <dd>{overview.model.revision}</dd>
        </div>
        <div>
          <dt>Device / dtype</dt>
          <dd>
            {overview.model.device} / {overview.model.dtype}
          </dd>
        </div>
        <div>
          <dt>Retained bytes</dt>
          <dd>{number(overview.summary.retained_bytes)}</dd>
        </div>
        <div>
          <dt>Capture / forward time</dt>
          <dd>
            {number(overview.summary.duration_ms)} /{" "}
            {number(overview.summary.forward_duration_ms)} ms
          </dd>
        </div>
        <div>
          <dt>Capture profile</dt>
          <dd>{overview.capture.profile}</dd>
        </div>
        <div>
          <dt>Expires at</dt>
          <dd>
            {trace.expires_at
              ? new Date(trace.expires_at).toLocaleString()
              : "Not provided"}
          </dd>
        </div>
      </dl>
      {overview.warnings.map((warning) => (
        <p key={warning} className="unavailable">
          {warning}
        </p>
      ))}
      <RawData
        label="Validation and runtime provenance"
        data={{
          validation: overview.validation,
          runtime: overview.runtime,
          capture: overview.capture,
        }}
      />
      <div className="detail-actions">
        <button className="secondary-button" onClick={onVerify}>
          Verify retained trace
        </button>
        <button
          className="text-button danger"
          onClick={onDelete}
          disabled={deleting}
        >
          {deleting ? "Deleting…" : "Delete trace"}
        </button>
      </div>
    </details>
  );
}
