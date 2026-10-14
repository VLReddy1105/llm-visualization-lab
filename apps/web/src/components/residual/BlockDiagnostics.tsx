import type { MLPInternalsResult, ResidualStreamResult } from "@/lib/api/client";
import { number, RawData } from "@/components/ui/ScientificData";

type Operation = ResidualStreamResult["operations"][number];
type Comparison = ResidualStreamResult["comparisons"][number];
export const scalar = (value: number | null | undefined) => value == null ? "Undefined" : number(value, 7);

export function OperationDetail({ operation }: { operation: Operation }) {
  const op = operation;
  return <div className="block-operation-detail" data-testid="block-operation-detail">
    <p className="eyebrow">SELECTED OPERATION · {op.symbol}</p>
    <h4>{op.name}</h4>
    <p className="block-formula"><code>{op.formula}</code></p>
    {op.symbol === "m" && <p>MLP Update: this vector was produced by applying the model’s down projection to the elementwise product of the activated gate and up projection.</p>}
    <dl className="facts">
      <div><dt>Input shape</dt><dd>{op.input_shapes.map((s) => s.join(" × ")).join("; ") || "Layer input"}</dd></div>
      <div><dt>Output shape</dt><dd>{op.output_shape.join(" × ")}</dd></div>
      <div><dt>Selected vector</dt><dd>{op.selected_shape.join(" × ")}</dd></div>
      <div><dt>Coordinate [{op.selected_coordinate.index}]</dt><dd>{scalar(op.selected_coordinate.value)}</dd></div>
      <div><dt>L2 norm</dt><dd>{scalar(op.statistics.l2_norm)}</dd></div>
      <div><dt>Mean / std</dt><dd>{scalar(op.statistics.mean)} / {scalar(op.statistics.std)}</dd></div>
      <div><dt>Minimum / maximum</dt><dd>{scalar(op.statistics.min)} / {scalar(op.statistics.max)}</dd></div>
      <div><dt>Evidence</dt><dd>{op.provenance === "raw" ? "Directly captured" : op.provenance} · {op.dtype} · {op.original_device}</dd></div>
    </dl>
    <p className="fine-print">{op.method} Statistics are backend summaries of the selected token, accumulated in {op.statistics.accumulation_dtype}.</p>
    <p className="fine-print"><code>{op.component_id}</code></p>
  </div>;
}

export function CoordinatePlot({ rows, series, label, signed = false }: {
  rows: { index: number; [key: string]: number }[];
  series: { key: string; label: string; color: string }[];
  label: string;
  signed?: boolean;
}) {
  // Chart geometry only: scientific values and coordinate selection come from Python.
  const magnitudes = rows.flatMap((row) => series.map((s) => Math.abs(row[s.key])));
  const extent = Math.max(...magnitudes, Number.MIN_VALUE);
  const y = (value: number) => 85 - (value / extent) * 66;
  const x = (index: number) => 28 + index * (540 / Math.max(rows.length - 1, 1));
  return <figure className="block-chart" aria-label={label}>
    <figcaption>{label}</figcaption>
    {!rows.length && <p className="fine-print">No coordinates match this sign filter.</p>}
    <svg viewBox="0 0 600 185" role="img" aria-label={`${label}. Exact values are in the coordinate table.`}>
      <line x1="22" y1="85" x2="578" y2="85" stroke="#bac8d8" />
      <text x="4" y="89">0</text>
      {series.map((s) => signed
        ? rows.map((row, i) => <rect key={`${s.key}-${row.index}`} x={x(i) - 6} y={Math.min(y(row[s.key]), 85)} width="12" height={Math.abs(y(row[s.key]) - 85)} fill={row[s.key] < 0 ? "#ad5264" : s.color}><title>Coordinate {row.index}: {scalar(row[s.key])}</title></rect>)
        : <g key={s.key}><polyline fill="none" stroke={s.color} strokeWidth="2" points={rows.map((row, i) => `${x(i)},${y(row[s.key])}`).join(" ")} />{rows.map((row, i) => <circle key={row.index} cx={x(i)} cy={y(row[s.key])} r="3" fill={s.color}><title>{s.label}, coordinate {row.index}: {scalar(row[s.key])}</title></circle>)}</g>)}
      {rows.filter((_, i) => rows.length <= 16 || i % Math.ceil(rows.length / 12) === 0).map((row) => <text key={row.index} x={x(rows.indexOf(row))} y="177" textAnchor="middle">{row.index}</text>)}
    </svg>
    <div className="block-legend">{series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>)}<span>Absolute coordinate indices · bounded server selection</span></div>
  </figure>;
}

export function ResidualDiagnostics({ comparison }: { comparison: Comparison }) {
  const c = comparison;
  return <>
    <dl className="block-metrics">
      <div><dt>Incoming residual L2</dt><dd>{scalar(c.before_norm)}</dd></div>
      <div><dt>Update L2</dt><dd>{scalar(c.update_norm)}</dd></div>
      <div><dt>After addition L2</dt><dd>{scalar(c.after_norm)}</dd></div>
      <div><dt>Delta magnitude</dt><dd>{scalar(c.delta_norm)}</dd></div>
      <div><dt>Cosine · before / after</dt><dd>{scalar(c.cosine_before_after)}</dd></div>
      <div><dt>Update / residual norm</dt><dd>{scalar(c.update_to_residual_ratio)}</dd></div>
    </dl>
    {Object.entries(c.undefined).map(([name, reason]) => <p className="unavailable" key={name}>{name}: undefined — {reason}</p>)}
    <CoordinatePlot rows={c.coordinates} series={[{ key: "before", label: c.before_symbol, color: "#6754a4" }, { key: "update", label: c.update_symbol, color: "#237e77" }, { key: "after", label: c.after_symbol, color: "#2457b2" }]} label="Before / update / after coordinates" />
    <CoordinatePlot rows={c.coordinates} series={[{ key: "signed_delta", label: "After − before (derived)", color: "#237e77" }]} label="Signed coordinate changes" signed />
    <p className="fine-print">Norms and deltas describe magnitude, not causal importance. Signed differences are derived from residual checkpoints; attention and MLP updates are directly captured branch outputs.</p>
  </>;
}

export function Reconciliation({ checks }: { checks: ResidualStreamResult["reconciliation"] | MLPInternalsResult["reconciliation"] }) {
  return <div className="block-reconciliation">
    <h4>Numerical reconciliation</h4>
    <div className="block-table-scroll"><table>
      <caption>Backend float32 comparison with the recorded validation tolerances</caption>
      <thead><tr><th>Equation / reference</th><th>Status</th><th>Max absolute error</th><th>Relative L2 error</th><th>Absolute / relative tolerance</th></tr></thead>
      <tbody>{checks.map((c) => <tr key={c.name}><td><code>{c.formula}</code><small className="block-check-scope">{c.provenance === "independent_reference" ? "Independent module reference" : "Derived from saved operands"} · {c.scope}</small></td><td><span className={c.metrics.passed ? "block-pass" : "validation-failed"}>{c.metrics.passed ? "PASS" : "FAIL"}</span></td><td>{scalar(c.metrics.max_absolute_error)}</td><td>{scalar(c.metrics.relative_l2_error)}</td><td>{scalar(c.metrics.absolute_tolerance)} / {scalar(c.metrics.relative_tolerance)}</td></tr>)}</tbody>
    </table></div>
    <p className="fine-print">Reconciliation is a derived comparison of captured operands and outputs. Relative L2 uses each check’s recorded denominator epsilon; zero-norm cosines and ratios remain undefined.</p>
    <RawData label="Reconciliation provenance and full metrics" data={checks} />
  </div>;
}
