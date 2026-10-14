import { useState } from "react";
import type { BlockParameters, MLPInternalsResult, TraceOverview } from "@/lib/api/client";
import { ApiError } from "@/lib/api/client";
import { useAttentionCapture } from "@/components/attention/useAttentionCapture";
import { useAnalysis } from "@/components/evolution/useAnalysis";
import { AnalysisStatus } from "@/components/evolution/AnalysisStatus";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { RawData } from "@/components/ui/ScientificData";
import { useSelection } from "@/state/selection";
import { BlockFlow } from "./BlockFlow";
import { CoordinatePlot, OperationDetail, Reconciliation, ResidualDiagnostics, scalar } from "./BlockDiagnostics";

export function BlockNavigation({ fallbackToken }: { fallbackToken: number }) {
  const open = useSelection((s) => s.openBlock);
  const active = useSelection((s) => s.selectedComponent === "residual" ? s.blockView : null);
  return <nav className="block-navigation" aria-label="Residual and MLP views">
    <span className="eyebrow">INSIDE THE BLOCK</span>
    <button className="secondary-button" aria-pressed={active === "residual"} onClick={() => open("residual", fallbackToken)}>Residual Stream</button>
    <button className="secondary-button" aria-pressed={active === "mlp"} onClick={() => open("mlp", fallbackToken)}>MLP Explorer</button>
  </nav>;
}

function CoordinateControls({ value, hiddenSize, intermediateSize, onApply }: {
  value: BlockParameters; hiddenSize: number; intermediateSize: number; onApply: (value: BlockParameters) => void;
}) {
  const [draft, setDraft] = useState(value);
  const [error, setError] = useState<string | null>(null);
  return <form className="block-coordinate-controls" onSubmit={(event) => {
    event.preventDefault();
    if (!Number.isInteger(draft.hidden_coordinate_index) || draft.hidden_coordinate_index! < 0 || draft.hidden_coordinate_index! >= hiddenSize
      || !Number.isInteger(draft.intermediate_coordinate_index) || draft.intermediate_coordinate_index! < 0 || draft.intermediate_coordinate_index! >= intermediateSize
      || !Number.isInteger(draft.top_k) || draft.top_k! < 1 || draft.top_k! > 64) {
      setError(`Use hidden coordinates 0–${hiddenSize - 1}, intermediate coordinates 0–${intermediateSize - 1}, and a count of 1–64.`);
      return;
    }
    setError(null);
    onApply(draft);
  }} noValidate>
    <label>Hidden coordinate<input type="number" min="0" max={hiddenSize - 1} value={draft.hidden_coordinate_index} onChange={(e) => setDraft({ ...draft, hidden_coordinate_index: e.target.valueAsNumber })} /></label>
    <label>Intermediate coordinate<input type="number" min="0" max={intermediateSize - 1} value={draft.intermediate_coordinate_index} onChange={(e) => setDraft({ ...draft, intermediate_coordinate_index: e.target.valueAsNumber })} /></label>
    <label>Ranking<select aria-label="Ranking" value={draft.ranking} onChange={(e) => setDraft({ ...draft, ranking: e.target.value as BlockParameters["ranking"] })}>
      <option value="absolute">Largest absolute values</option><option value="positive">Largest positive coordinates</option><option value="negative">Largest negative coordinates</option>
    </select></label>
    <label>Coordinate count<input type="number" min="1" max="64" value={draft.top_k} onChange={(e) => setDraft({ ...draft, top_k: e.target.valueAsNumber })} /></label>
    <button className="secondary-button" type="submit">Apply coordinates</button>
    {error && <p className="block-input-error" role="alert">{error}</p>}
  </form>;
}

function GateCoordinates({ result }: { result: MLPInternalsResult }) {
  const row = result.selected_gate_coordinate;
  const before = result.operations.find((op) => op.symbol === "r1")!.selected_coordinate;
  const update = result.operations.find((op) => op.symbol === "m")!.selected_coordinate;
  const after = result.operations.find((op) => op.symbol === "r2")!.selected_coordinate;
  return <div className="gate-coordinates">
    <p className="eyebrow">EXACT INTERMEDIATE COORDINATE {row.index}</p>
    <h4>Gate × up, after SiLU</h4>
    <p className="block-formula"><code>product[{row.index}] = SiLU(gate[{row.index}]) × up[{row.index}]</code></p>
    <dl className="block-metrics">
      <div><dt>Raw gate</dt><dd>{scalar(row.gate)}</dd></div>
      <div><dt>SiLU(gate)</dt><dd>{scalar(row.activated_gate)}</dd></div>
      <div><dt>Up projection</dt><dd>{scalar(row.up)}</dd></div>
      <div><dt>Product</dt><dd>{scalar(row.product)}</dd></div>
      <div><dt>Multiplication error</dt><dd>{scalar(row.multiplication_error)}</dd></div>
    </dl>
    <div className="block-table-scroll"><table>
      <caption>Intermediate coordinates ranked by the product, with shared absolute indices</caption>
      <thead><tr><th>Coordinate</th><th>Gate</th><th>SiLU(gate)</th><th>Up</th><th>Product</th></tr></thead>
      <tbody>{result.gate_coordinates.map((c) => <tr key={c.index}><td>{c.index}</td><td>{scalar(c.gate)}</td><td>{scalar(c.activated_gate)}</td><td>{scalar(c.up)}</td><td>{scalar(c.product)}</td></tr>)}</tbody>
    </table></div>
    <p className="fine-print">The product has {result.intermediate_size} intermediate coordinates. Down projection maps it into {result.source.hidden_size} hidden coordinates. Coordinate indices do not imply named concepts.</p>
    <p className="block-formula"><code>Residual addition [{after.index}]: r1 = {scalar(before.value)}; m = {scalar(update.value)}; r2 = {scalar(after.value)}. r2 = r1 + m.</code></p>
    <p className="fine-print">These three hidden-coordinate values are directly captured from the incoming MLP residual, down projection and residual output. The numerical reconciliation below checks the complete vectors.</p>
  </div>;
}

function BlockAnalysis({ id, token, layer, view, overview, onCapture, capturing }: { id: string; token: number; layer: number; view: "residual" | "mlp"; overview: TraceOverview; onCapture: () => void; capturing: boolean }) {
  const [parameters, setParameters] = useState<BlockParameters>({ token_index: token, layer_index: layer, top_k: 12, ranking: "absolute", hidden_coordinate_index: 0, intermediate_coordinate_index: 0 });
  const operation = useSelection((s) => s.selectedOperation);
  const selectOperation = useSelection((s) => s.selectOperation);
  const kind = view === "residual" ? "residual_stream" : "mlp_internals";
  const analysis = useAnalysis(id, { kind, parameters });
  const result = analysis.ready ? analysis.data?.result : null;
  const blockResult = result?.kind === "residual_stream" || result?.kind === "mlp_internals" ? result : null;
  const selected = operation && (blockResult?.operations.some((op) => op.symbol === operation) || operation === "output") ? operation : view === "residual" ? "r1" : "p";
  const op = blockResult?.operations.find((o) => o.symbol === (selected === "output" ? "r2" : selected));
  const comparison = blockResult?.kind === "residual_stream" ? blockResult.comparisons.find((c) => c.name === (["n_mlp", "m", "r2", "output"].includes(selected) ? "mlp" : "attention")) : null;
  const mismatched = !!blockResult && (blockResult.source.trace_id !== id || blockResult.layer_index !== layer || blockResult.source.token.token_index !== token || blockResult.kind !== kind || analysis.data?.trace_id !== id);
  return <>
    <CoordinateControls key={`${id}-${token}-${layer}`} value={parameters} hiddenSize={overview.model.architecture.hidden_size} intermediateSize={overview.model.architecture.intermediate_size} onApply={setParameters} />
    {mismatched ? <ApiErrorNotice error={new ApiError("The analysis source does not match the selected trace, layer and token. No scientific values are displayed.", "ANALYSIS_SOURCE_MISMATCH")} />
      : !blockResult || !op ? <><AnalysisStatus analysis={analysis} />{analysis.error instanceof ApiError && ["COMPONENT_NOT_CAPTURED", "TRACE_EXPIRED", "TRACE_DELETED", "TRACE_NOT_FOUND", "TRACE_EVICTED"].includes(analysis.error.code) && <button className="primary-button" disabled={capturing} onClick={onCapture}>Capture Layer {layer} Token {token} Details</button>}</>
        : <div className="block-analysis" data-testid="block-analysis" data-trace-id={id} data-layer-index={layer} data-token-index={token} data-kind={kind}>
          <div className="block-context"><div><p className="eyebrow">LAYER {layer} · INPUT POSITION {token}</p><h4>{JSON.stringify(blockResult.source.token.decoded_token)} <small>token ID {blockResult.source.token.token_id}</small></h4></div><span className="evidence-label">Captured vectors · derived diagnostics · {blockResult.source.device} / {blockResult.source.dtype}</span></div>
          <BlockFlow operations={blockResult.operations} view={view} selected={selected} onSelect={selectOperation} />
          <div className="block-detail-grid">
            <OperationDetail operation={op} />
            <div>{comparison ? <ResidualDiagnostics comparison={comparison} /> : <CoordinatePlot rows={op.coordinates} series={[{ key: "value", label: op.symbol, color: "#237e77" }]} label={`${op.name}: signed coordinate values`} signed />}</div>
          </div>
          {blockResult.kind === "mlp_internals" && <GateCoordinates result={blockResult} />}
          <div className="block-table-scroll"><table>
            <caption>{comparison ? `${comparison.name} branch: coordinates ranked by ${parameters.ranking} update` : `${op.name}: bounded ${parameters.ranking} coordinate ranking`}</caption>
            <thead><tr><th>Coordinate</th>{comparison ? <><th>Before</th><th>Branch update</th><th>After</th><th>Signed difference (derived)</th></> : <th>Signed value</th>}</tr></thead>
            <tbody>{(comparison ? comparison.coordinates : op.coordinates).map((row) => <tr key={row.index}><td>{row.index}</td>{"value" in row ? <td>{scalar(row.value)}</td> : <><td>{scalar(row.before)}</td><td>{scalar(row.update)}</td><td>{scalar(row.after)}</td><td>{scalar(row.signed_delta)}</td></>}</tr>)}</tbody>
          </table></div>
          <p className="fine-print">{blockResult.ranking_policy}</p>
          <Reconciliation checks={blockResult.reconciliation} />
          <details className="analysis-provenance"><summary>Source identity, capture provenance and analysis method</summary>
            <dl className="facts"><div><dt>Source trace</dt><dd>{blockResult.source.trace_id}</dd></div><div><dt>Analysis</dt><dd>{analysis.data?.analysis_id} · {analysis.data?.method_version}</dd></div><div><dt>Revision</dt><dd>{blockResult.source.revision}</dd></div><div><dt>Token / layer</dt><dd>{token} / {layer}</dd></div><div><dt>Retention</dt><dd>{analysis.data?.retained_bytes} bytes · expires {analysis.data?.expires_at}</dd></div><div><dt>Cache</dt><dd>{analysis.data?.cache_hit ? "Reused immutable analysis" : "Computed from retained values"}</dd></div></dl>
            <p className="fine-print">{blockResult.method}</p><RawData data={blockResult} label="Bounded scientific result" />
          </details>
        </div>}
  </>;
}

export function BlockWorkspace({ sourceId, source }: { sourceId: string; source: TraceOverview }) {
  const layer = useSelection((s) => s.selectedLayer) ?? 0;
  const token = useSelection((s) => s.selectedToken) ?? source.input.sequence_length - 1;
  const view = useSelection((s) => s.blockView);
  const capture = useAttentionCapture(sourceId, layer, source, "mlp", token);
  const failure = capture.issue ?? capture.capture.error ?? capture.trace.error ?? capture.cancel.error
    ?? (capture.trace.data?.status === "failed" || capture.trace.data?.status === "cancelled"
      ? new ApiError(String(capture.trace.data.error?.message ?? "Detailed block capture did not complete."), String(capture.trace.data.error?.code ?? "TRACE_CAPTURE_FAILED")) : null);
  if (failure || !capture.ready) return <div className="capture-block">
    <p className="eyebrow">SELECTED LAYER {layer} · TOKEN {token}</p>
    {failure && <ApiErrorNotice error={failure} />}
    {capture.pending && !failure ? <><p role="status">Capturing selected residual and MLP tensors…</p><button className="text-button" onClick={() => capture.cancel.mutate()}>Cancel capture</button>{capture.timedOut && <button className="secondary-button" onClick={() => void capture.trace.refetch()}>Check capture status</button>}</>
      : <><h4>Detailed block data was not captured for this selection.</h4><p>Capture layer {layer}, token {token} from the same raw prompt. This explicit model forward creates an independently identified trace with the residual, attention output and gated MLP tensors. Subsequent exploration reads retained data.</p><button className="primary-button" disabled={capture.capture.isPending} onClick={() => capture.capture.mutate()}>Capture Layer {layer} Token {token} Details</button></>}
  </div>;
  return <div className="block-workspace">
    <p className="capture-lineage" data-testid="block-source">Source trace: {capture.id}.{capture.id !== sourceId && <> Separate detailed execution. Original trace: {sourceId}. Every value below belongs to the detailed trace.</>}</p>
    <BlockAnalysis key={`${capture.id}-${token}-${layer}`} id={capture.id!} token={token} layer={layer} view={view} overview={capture.trace.data!.overview!} onCapture={() => capture.capture.mutate()} capturing={capture.capture.isPending} />
  </div>;
}
