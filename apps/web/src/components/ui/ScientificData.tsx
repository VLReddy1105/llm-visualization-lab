import type { ComponentDescriptor } from "@/lib/api/client";

export const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
export const number = (value: number | null | undefined) =>
  value == null
    ? "Unavailable"
    : value.toLocaleString("en", { maximumFractionDigits: 3 });
export const visibleWhitespace = (text: string) =>
  text
    .replaceAll(" ", "·")
    .replaceAll("\n", "↵")
    .replaceAll("\t", "⇥")
    .replaceAll("\r", "␍");
export function RawData({
  data,
  label = "Raw descriptor",
}: {
  data: unknown;
  label?: string;
}) {
  return (
    <details className="raw-data">
      <summary>{label}</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}
export function Descriptor({ data }: { data: ComponentDescriptor }) {
  return (
    <div className="descriptor">
      <span className="evidence-label">
        {data.provenance === "raw" ? "Captured" : data.provenance} ·{" "}
        {data.raw_available ? "retained tensor" : "summary only"}
      </span>
      <dl className="facts">
        <div>
          <dt>Shape</dt>
          <dd>{data.logical_shape.join(" × ")}</dd>
        </div>
        <div>
          <dt>Axes</dt>
          <dd>{data.axes.join(" · ")}</dd>
        </div>
        <div>
          <dt>Precision</dt>
          <dd>
            {data.dtype} · {data.original_device}
          </dd>
        </div>
        {data.statistics && (
          <>
            <div>
              <dt>L2 norm</dt>
              <dd>{number(data.statistics.l2_norm)}</dd>
            </div>
            <div>
              <dt>Mean / std</dt>
              <dd>
                {number(data.statistics.mean)} / {number(data.statistics.std)}
              </dd>
            </div>
          </>
        )}
      </dl>
      {data.statistics && (
        <p className="fine-print">
          Statistics: {data.statistics.reduction_scope} ·{" "}
          {data.statistics.accumulation_dtype} accumulation.
        </p>
      )}
      <p className="fine-print">{data.method}</p>
      {!data.raw_available && (
        <p className="unavailable">Not captured in this trace · raw values</p>
      )}
      <RawData data={data} />
    </div>
  );
}
