import type { KeyboardEvent } from "react";
import type { Checkpoint } from "@/lib/api/client";
import { number } from "@/components/ui/ScientificData";

type Series = { name: string; values: (number | null)[] };
const colors = [
  "#235c9c",
  "#0b776c",
  "#8056a2",
  "#956012",
  "#a03c55",
  "#596875",
  "#584ba0",
  "#6b6a14",
];
export function DepthChart({
  title,
  unit,
  checkpoints,
  series,
  selected,
  onSelect,
  rank = false,
}: {
  title: string;
  unit: string;
  checkpoints: Checkpoint[];
  series: Series[];
  selected: number;
  onSelect: (index: number) => void;
  rank?: boolean;
}) {
  // Presentation coordinates only: no scientific reductions or normalization here.
  const values = series.flatMap((s) =>
    s.values.filter((v): v is number => v != null),
  );
  if (!values.length) return <p>{title}: no defined values.</p>;
  const min = rank ? Math.min(...values) : Math.min(0, ...values);
  const max = Math.max(...values),
    span = max - min || 1;
  const x = (i: number) => 66 + (i / Math.max(1, checkpoints.length - 1)) * 840;
  const y = (v: number) =>
    rank ? 30 + ((v - min) / span) * 168 : 198 - ((v - min) / span) * 168;
  function keyboard(
    event: KeyboardEvent<SVGCircleElement>,
    index: number,
    seriesIndex: number,
  ) {
    if (["Enter", " "].includes(event.key)) {
      event.preventDefault();
      onSelect(index);
    }
    const target =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? checkpoints.length - 1
          : event.key === "ArrowRight"
            ? Math.min(index + 1, checkpoints.length - 1)
            : event.key === "ArrowLeft"
              ? Math.max(0, index - 1)
              : null;
    if (target != null) {
      event.preventDefault();
      onSelect(target);
      event.currentTarget
        .closest("svg")
        ?.querySelector<SVGCircleElement>(
          `[data-series="${seriesIndex}"][data-depth="${target}"]`,
        )
        ?.focus();
    }
  }
  return (
    <section className="depth-chart" aria-label={title}>
      <div className="section-heading">
        <h4>{title}</h4>
        <span className="fine-print">{unit}</span>
      </div>
      <div className="chart-scroll">
        <svg
          viewBox="0 0 950 244"
          role="group"
          aria-label={`${title}; select a point with Enter; arrow keys move through depth`}
        >
          {[0, 0.5, 1].map((fraction) => {
            const value = min + fraction * span;
            return (
              <g key={fraction}>
                <line
                  x1="66"
                  x2="906"
                  y1={y(value)}
                  y2={y(value)}
                  stroke="#dfe5ed"
                />
                <text x="58" y={y(value) + 4} textAnchor="end">
                  {number(value, 3)}
                </text>
              </g>
            );
          })}
          <line
            x1={x(selected)}
            x2={x(selected)}
            y1="22"
            y2="204"
            stroke="#8798ad"
            strokeDasharray="3 4"
          />
          {series.map((s, si) => (
            <g key={s.name}>
              {s.values.map((v, i) =>
                v != null && i > 0 && s.values[i - 1] != null ? (
                  <line
                    key={`line-${i}`}
                    x1={x(i - 1)}
                    y1={y(s.values[i - 1]!)}
                    x2={x(i)}
                    y2={y(v)}
                    stroke={colors[si % colors.length]}
                    strokeWidth="2"
                  />
                ) : null,
              )}
              {s.values.map((v, i) =>
                v == null ? null : (
                  <circle
                    key={i}
                    data-series={si}
                    data-depth={i}
                    cx={x(i)}
                    cy={y(v)}
                    r={i === selected ? 5 : 3.5}
                    fill={colors[si % colors.length]}
                    role="button"
                    tabIndex={i === selected ? 0 : -1}
                    aria-label={`${s.name}, ${checkpoints[i].label}: ${v}`}
                    onClick={() => onSelect(i)}
                    onKeyDown={(e) => keyboard(e, i, si)}
                  >
                    <title>
                      {s.name} · {checkpoints[i].label} · {v}
                    </title>
                  </circle>
                ),
              )}
            </g>
          ))}
          {checkpoints.map((c, i) =>
            i === 0 ||
            i === checkpoints.length - 1 ||
            (i > 2 && (i - 1) % 5 === 0) ? (
              <text key={c.component_id} x={x(i)} y="226" textAnchor="middle">
                {c.stage === "embedding"
                  ? "Embed"
                  : c.stage === "final_normalized"
                    ? "Norm"
                    : `L${c.layer_index}`}
              </text>
            ) : null,
          )}
        </svg>
      </div>
      <ul className="chart-legend">
        {series.map((s, i) => (
          <li key={s.name}>
            <span
              style={{ background: colors[i % colors.length] }}
              aria-hidden="true"
            />
            {s.name}{" "}
            <strong>
              {s.values[selected] == null
                ? "Undefined"
                : number(s.values[selected], 6)}
            </strong>
          </li>
        ))}
      </ul>
    </section>
  );
}
