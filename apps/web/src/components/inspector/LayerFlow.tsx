import { useState } from "react";
import type { ComponentGroup } from "@/lib/api/client";
import { useSelection } from "@/state/selection";
import { Descriptor } from "@/components/ui/ScientificData";

const detailed = [
  ["residual.before", "Residual before"],
  ["attention.input_normalized", "RMSNorm"],
  ["projections", "q_proj / k_proj / v_proj"],
  ["qk_norm", "q_norm / k_norm"],
  ["rope", "RoPE · Q & K"],
  ["attention", "Attention"],
  ["attention.output", "o_proj"],
  ["residual.after_attention", "+ Residual"],
  ["mlp.input_normalized", "RMSNorm"],
  ["mlp.output", "Gated MLP"],
  ["residual.after_mlp", "+ Residual"],
  ["output", "Layer output"],
];
const simple = [
  ["residual.before", "Residual"],
  ["attention", "Attention"],
  ["residual.after_attention", "+ Add"],
  ["mlp.output", "MLP"],
  ["residual.after_mlp", "+ Add"],
];
export function LayerFlow({
  index,
  fallbackToken,
  data,
  insideAttention = false,
}: {
  index: number;
  fallbackToken: number;
  data?: ComponentGroup;
  insideAttention?: boolean;
}) {
  const [expanded, expand] = useState(false);
  const operation = useSelection((s) => s.selectedOperation);
  const select = useSelection((s) => s.selectOperation);
  const attention = useSelection((s) => s.selectAttention);
  const openBlock = useSelection((s) => s.openBlock);
  const descriptor = data?.components.find(
    (d) => d.component_id === `layer[${index}].${operation}`,
  );
  return (
    <div className="layer-microscope">
      <div className="section-heading">
        <div>
          <p className="eyebrow">LAYER MICROSCOPE</p>
          <p>Two residual additions. One continuous representation.</p>
        </div>
        <button
          className="secondary-button"
          aria-pressed={expanded}
          onClick={() => expand(!expanded)}
        >
          {expanded ? "Simple view" : "Detailed view"}
        </button>
      </div>
      <div
        className="microscope-scroll"
        tabIndex={0}
        aria-label={`Layer ${index} computation, horizontally scrollable`}
      >
        <div className="microscope-flow">
          {(expanded ? detailed : simple).map(([id, label], i) => (
            <div className="microscope-step" key={id}>
              {i > 0 && (
                <span className="flow-arrow" aria-hidden="true">
                  →
                </span>
              )}
              <button
                className={`microscope-node ${id === "attention" ? "attention-node" : ""}`}
                aria-pressed={
                  (id === "attention" && insideAttention) || operation === id
                }
                onClick={() => {
                  if (id === "attention") attention(index, fallbackToken);
                  else if (id.startsWith("mlp")) openBlock("mlp", fallbackToken, index);
                  else if (id.startsWith("residual")) openBlock("residual", fallbackToken, index);
                  else select(operation === id ? null : id);
                }}
              >
                <small>{String(i + 1).padStart(2, "0")}</small>
                {label}
              </button>
            </div>
          ))}
        </div>
      </div>
      <p className="fine-print">
        {expanded
          ? "V bypasses Q/K normalization and RoPE. Residual input bypasses attention to the first add; that sum bypasses the gated MLP to the second add."
          : "Residual paths bypass attention and MLP into their respective additions. Expand for Qwen3’s Q/K normalization and rotary operations."}
      </p>
      {operation && (
        <div className="operation-detail">
          {descriptor ? (
            <Descriptor data={descriptor} />
          ) : (
            <p>
              {operation === "projections"
                ? "Separate learned projections create Q, K and V. Inspect their selected vectors inside Attention."
                : operation === "qk_norm"
                  ? "Qwen3 normalizes Q and K separately along each head’s dimensions before applying RoPE."
                  : operation === "rope"
                    ? "Rotary position encoding transforms Q and K using the retained position-specific sine and cosine values. V is unchanged."
                    : "Detailed values for this operation were not captured in this trace."}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
