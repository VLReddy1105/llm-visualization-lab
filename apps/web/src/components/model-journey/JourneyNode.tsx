import type { JourneyComponent } from "@/state/selection";

export type JourneyItem = {
  key: string;
  label: string;
  category: "neutral" | "residual" | "transformer" | "output";
  component: JourneyComponent;
  layer?: number;
  detail?: string;
};
export function JourneyNode({
  item,
  selected,
  enabled,
  onSelect,
}: {
  item: JourneyItem;
  selected: boolean;
  enabled: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`journey-node ${item.category} ${selected ? "selected" : ""} ${item.component === "prediction" ? "prediction-node" : ""}`}
      data-node={item.key}
      aria-label={item.layer != null ? `Layer ${item.layer}` : item.label}
      aria-pressed={selected}
      disabled={!enabled}
      onClick={onSelect}
    >
      <span className="node-symbol" aria-hidden="true">
        {item.layer != null ? (
          <>
            <i />
            <i />
            <i />
          </>
        ) : item.component === "input" ? (
          "Aa"
        ) : item.component === "tokens" ? (
          "[ ]"
        ) : item.component === "embedding.output" ? (
          "≋"
        ) : item.component === "output.normalized" ? (
          "∥"
        ) : item.component === "output.lm_head_input" ? (
          "↗"
        ) : item.component === "logits" ? (
          "▥"
        ) : (
          "→"
        )}
      </span>
      <span className="node-label">{item.label}</span>
      {item.detail && <span className="node-detail">{item.detail}</span>}
    </button>
  );
}
