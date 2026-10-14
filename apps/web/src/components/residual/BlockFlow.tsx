import { useRef } from "react";
import type { MLPInternalsResult, ResidualStreamResult } from "@/lib/api/client";
import { number } from "@/components/ui/ScientificData";

type Operation = ResidualStreamResult["operations"][number];
const residualSteps = [
  ["r0", "Residual input", "Before attention"],
  ["n_attn", "RMSNorm", "Attention input"],
  ["a", "Attention update", "After o_proj"],
  ["r1", "+ Residual", "After attention"],
  ["n_mlp", "MLP RMSNorm", "Before MLP"],
  ["m", "MLP update", "After down_proj"],
  ["r2", "+ Residual", "After MLP"],
  ["output", "Residual output", "Next layer"],
];
const mlpSteps = [
  ["n_mlp", "Normalized input", "Learned RMSNorm"],
  ["g", "Gate projection", "gate_proj"],
  ["u", "Up projection", "up_proj"],
  ["s", "SiLU", "g · sigmoid(g)"],
  ["p", "Multiply", "s ⊙ u"],
  ["m", "Down projection", "MLP update"],
  ["r2", "+ Residual", "r1 + m"],
];

export function BlockFlow({ operations, view, selected, onSelect }: {
  operations: (Operation | MLPInternalsResult["operations"][number])[];
  view: "residual" | "mlp";
  selected: string;
  onSelect: (symbol: string) => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const steps = view === "residual" ? residualSteps : mlpSteps;
  return <div className="block-flow-scroll" role="region" aria-label={view === "residual" ? "Residual computation flow" : "Gated MLP computation flow"} tabIndex={0}>
    <div ref={root} className={`block-flow ${view}-flow`} role="toolbar" aria-label="Select a computation operation">
      {view === "mlp" && <svg className="mlp-connectors" viewBox="0 0 1000 210" preserveAspectRatio="none" aria-hidden="true">
        <path d="M125 105 H153 V50 H190 M153 105 V160 H190 M315 50 H360 M315 160 H530 V105 H550 M485 50 H530 V105 H550 M670 105 H715 M835 105 H875" />
      </svg>}
      {steps.map(([symbol, label, caption], i) => {
        const op = operations.find((v) => v.symbol === (symbol === "output" ? "r2" : symbol));
        return <button key={symbol} data-operation={symbol} className={`block-operation operation-${symbol}`}
          aria-pressed={selected === symbol} tabIndex={selected === symbol ? 0 : -1}
          onClick={() => onSelect(symbol)}
          onKeyDown={(event) => {
            const index = event.key === "ArrowRight" ? Math.min(i + 1, steps.length - 1)
              : event.key === "ArrowLeft" ? Math.max(i - 1, 0)
                : event.key === "Home" ? 0 : event.key === "End" ? steps.length - 1 : null;
            if (index == null) return;
            event.preventDefault();
            onSelect(steps[index][0]);
            root.current?.querySelector<HTMLButtonElement>(`[data-operation="${steps[index][0]}"]`)?.focus();
          }}>
          <small>{caption}</small><strong>{label}</strong>
          <span>{op ? `‖${op.symbol}‖ ${number(op.statistics.l2_norm, 5)}` : "Not captured"}</span>
        </button>;
      })}
    </div>
    <p className="fine-print">{view === "residual"
      ? "r0 bypasses the attention branch into the first addition; r1 bypasses the MLP into the second addition. Arrow keys inspect neighboring operations."
      : "Gate and up projections receive the same normalized input. SiLU is g × sigmoid(g); its result multiplies up elementwise. The residual r1 enters the final addition."}</p>
  </div>;
}
