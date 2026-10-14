import { useState } from "react";
import type { EvolutionResult } from "@/lib/api/client";
import { number, RawData } from "@/components/ui/ScientificData";
import { DepthChart } from "./DepthChart";

export function RepresentationView({
  data,
  selected,
  onSelect,
  expanded,
}: {
  data: EvolutionResult;
  selected: number;
  onSelect: (index: number) => void;
  expanded: boolean;
}) {
  const [compareFrom, setCompareFrom] = useState(0);
  const points = data.checkpoints,
    active = points[selected];
  const a = Math.min(compareFrom, selected),
    b = Math.max(compareFrom, selected);
  const comparison = data.comparisons.find(
    (c) => c.from_checkpoint === a && c.to_checkpoint === b,
  );
  const chart = { checkpoints: points, selected, onSelect };
  return (
    <>
      <DepthChart
        {...chart}
        title="Hidden-State Norm Evolution"
        unit="L2 norm · float64"
        series={[
          { name: "Hidden state", values: points.map((c) => c.l2_norm) },
        ]}
      />
      <section
        className="checkpoint-inspector"
        aria-label="Selected hidden state"
      >
        <div>
          <p className="eyebrow">{active.stage.replaceAll("_", " ")}</p>
          <h4>{active.label}</h4>
          <code>{active.component_id}</code>
        </div>
        <dl className="vector-statistics">
          {(
            [
              ["L2 norm", active.l2_norm],
              ["Change in norm", active.norm_change],
              ["L2 distance from previous", active.consecutive_distance],
              ["Cosine with previous", active.cosine_previous],
              ["Cosine with embedding", active.cosine_embedding],
              ["Cosine with final raw residual", active.cosine_final_residual],
            ] as const
          ).map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd title={String(value)}>
                {value == null ? "Undefined" : number(value, 7)}
              </dd>
            </div>
          ))}
        </dl>
        {Object.entries(active.undefined).map(([key, reason]) => (
          <p className="fine-print" key={key}>
            {key.replaceAll("_", " ")}: {reason}.
          </p>
        ))}
        <details>
          <summary>Selected-vector statistics and exact metrics</summary>
          <RawData label="Exact checkpoint metrics" data={active} />
          <p className="fine-print">
            {active.statistics.reduction_scope}; collector statistics accumulate
            in {active.statistics.accumulation_dtype}.
          </p>
        </details>
      </section>
      {expanded && (
        <>
          <DepthChart
            {...chart}
            title="Consecutive-Layer Change"
            unit="L2 distance / signed norm change"
            series={[
              {
                name: "L2 distance",
                values: points.map((c) => c.consecutive_distance),
              },
              {
                name: "Change in norm",
                values: points.map((c) => c.norm_change),
              },
            ]}
          />
          <DepthChart
            {...chart}
            title="Cosine Similarity Through Depth"
            unit="Cosine · undefined at zero norm"
            series={[
              {
                name: "Previous",
                values: points.map((c) => c.cosine_previous),
              },
              {
                name: "Embedding",
                values: points.map((c) => c.cosine_embedding),
              },
              {
                name: "Final raw residual",
                values: points.map((c) => c.cosine_final_residual),
              },
            ]}
          />
        </>
      )}
      <section className="layer-comparison" aria-label="Layer Comparison">
        <h4>Layer Comparison</h4>
        <label>
          Compare active checkpoint with{" "}
          <select
            value={compareFrom}
            onChange={(e) => setCompareFrom(Number(e.target.value))}
          >
            {points.map((c, i) => (
              <option value={i} key={c.component_id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        {comparison && (
          <>
            <p>
              {points[a].label} → {points[b].label} · same token, same trace
            </p>
            <dl className="vector-statistics">
              <div>
                <dt>L2 distance</dt>
                <dd>{number(comparison.l2_distance, 7)}</dd>
              </div>
              <div>
                <dt>Change in norm</dt>
                <dd>{number(comparison.norm_change, 7)}</dd>
              </div>
              <div>
                <dt>Cosine similarity</dt>
                <dd>
                  {comparison.cosine_similarity == null
                    ? `Undefined: ${comparison.undefined_reason}`
                    : number(comparison.cosine_similarity, 7)}
                </dd>
              </div>
            </dl>
          </>
        )}
      </section>
      <p className="science-caveat">
        Norms and cosine similarities describe vector magnitude and direction.
        They do not establish semantic meaning or causal importance. Final norm
        is a separate transformation, not another residual block.
      </p>
      <details>
        <summary>Metric definitions and precision</summary>
        <p>{data.method}</p>
        <p>
          Cosine uses the full hidden vector; zero norms are undefined. The
          final-state reference is the last pre-normalization residual.
        </p>
      </details>
    </>
  );
}
