import { useQuery } from "@tanstack/react-query";
import type { TraceOverview } from "@/lib/api/client";
import { layerQuery } from "@/lib/api/queries";
import { number, RawData } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";

import { LayerFlow } from "./LayerFlow";

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
export function LayerInspector({
  id,
  index,
  overview,
}: {
  id: string;
  index: number;
  overview: TraceOverview;
}) {
  const query = useQuery(layerQuery(id, index));
  if (query.isPending)
    return <p role="status">Loading Layer {index} from this trace…</p>;
  if (query.error)
    return (
      <ApiErrorNotice
        error={query.error}
        onRetry={() => void query.refetch()}
      />
    );
  const data = query.data;
  const checks = Object.values(data.validation);
  const passed = checks.filter(
    (check) =>
      record(check) && (check.passed === true || check.status === "passed"),
  ).length;
  const failed = checks.filter(
    (check) =>
      record(check) && (check.passed === false || check.status === "failed"),
  ).length;
  const points = [
    { suffix: "residual.before", name: "Residual input" },
    { suffix: "attention.output", name: "Attention update" },
    { suffix: "residual.after_attention", name: "After attention" },
    { suffix: "mlp.output", name: "MLP update" },
    { suffix: "residual.after_mlp", name: "Layer output" },
  ];
  return (
    <div className="layer-inspector">
      <div className="inspector-intro">
        <p>Two transformations update the same residual representation.</p>
        <span
          className={`validation-status ${failed ? "validation-failed" : ""}`}
        >
          {failed
            ? `${failed} checks failed`
            : passed
              ? `${passed} retained checks passed`
              : "Validation not recorded"}
        </span>
      </div>
      <LayerFlow
        index={index}
        fallbackToken={overview.input.sequence_length - 1}
        data={data}
      />
      <p className="fine-print">
        Qwen3 layer microscope · summaries below cover all captured token
        positions.
      </p>
      <div className="layer-data">
        <div className="checkpoint-table">
          <table>
            <caption>Captured residual checkpoints and updates</caption>
            <thead>
              <tr>
                <th>Checkpoint</th>
                <th>Shape</th>
                <th>L2 norm</th>
                <th>Evidence</th>
              </tr>
            </thead>
            <tbody>
              {points.map((point) => {
                const descriptor = data.components.find(
                  (component) =>
                    component.component_id ===
                    `layer[${index}].${point.suffix}`,
                );
                return (
                  <tr key={point.suffix}>
                    <td>{point.name}</td>
                    <td>
                      {descriptor?.logical_shape.join(" × ") ?? "Not captured"}
                    </td>
                    <td>{number(descriptor?.statistics?.l2_norm)}</td>
                    <td>
                      {descriptor
                        ? descriptor.raw_available
                          ? "Captured + summary"
                          : "Summary only"
                        : "Not captured in this trace"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="capture-coverage">
          <h4>Capture coverage</h4>
          <p>
            <strong>Attention detail</strong>
            <span>
              {overview.available.attention_layers.includes(index)
                ? "Captured"
                : "Not captured in this trace"}
            </span>
          </p>
          <p>
            <strong>MLP detail</strong>
            <span>
              {overview.available.mlp_layers.includes(index)
                ? "Captured"
                : "Not captured in this trace"}
            </span>
          </p>
          <p className="fine-print">
            Selection reads the retained trace. Deeper capture requires an
            explicit new trace.
          </p>
        </div>
      </div>
      <RawData
        label="Layer structure, descriptors and reconciliation"
        data={data}
      />
    </div>
  );
}
