import {
  useEffect,
  useRef,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import { useQuery } from "@tanstack/react-query";
import type { TraceOverview } from "@/lib/api/client";
import { layersQuery } from "@/lib/api/queries";
import { useSelection } from "@/state/selection";
import { percent, visibleWhitespace } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { JourneyNode, type JourneyItem } from "./JourneyNode";

const start: JourneyItem[] = [
  { key: "input", label: "Input", category: "neutral", component: "input" },
  { key: "tokens", label: "Tokens", category: "neutral", component: "tokens" },
  {
    key: "embedding",
    label: "Embedding",
    category: "residual",
    component: "embedding.output",
  },
];
const end: JourneyItem[] = [
  {
    key: "norm",
    label: "Final Norm",
    category: "residual",
    component: "output.normalized",
  },
  {
    key: "head",
    label: "LM Head",
    category: "output",
    component: "output.lm_head_input",
  },
  { key: "logits", label: "Logits", category: "output", component: "logits" },
  {
    key: "prediction",
    label: "Next token",
    category: "output",
    component: "prediction",
  },
];

export function ModelJourney({
  id,
  overview,
  busy,
}: {
  id: string | null;
  overview?: TraceOverview | null;
  busy: boolean;
}) {
  const component = useSelection((state) => state.selectedComponent);
  const layer = useSelection((state) => state.selectedLayer);
  const depth = useSelection((state) => state.selectedDepth);
  const token = useSelection((state) => state.selectedToken);
  const mode = useSelection((state) => state.viewport);
  const setMode = useSelection((state) => state.setViewport);
  const selectLayer = useSelection((state) => state.selectLayer);
  const selectComponent = useSelection((state) => state.selectComponent);
  const viewport = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; left: number } | null>(null);
  const layers = useQuery({ ...layersQuery(id ?? ""), enabled: !!id });
  const indices = overview?.available.layers ?? [];
  const candidate = overview?.prediction.top_k[0];
  const tail = end.map((item) =>
    item.component === "prediction" && candidate
      ? {
          ...item,
          label: visibleWhitespace(candidate.decoded_token),
          detail: percent(candidate.probability),
        }
      : item,
  );
  const layerItems: JourneyItem[] = indices.map((index) => ({
    key: `layer-${index}`,
    label: `L${index}`,
    category: "transformer",
    component: "layer",
    layer: index,
  }));
  const items: JourneyItem[] = [...start, ...layerItems, ...tail];
  const activeKey =
    component === "evolution"
      ? depth === 0
        ? "embedding"
        : layer != null
          ? `layer-${layer}`
          : depth != null
            ? "norm"
            : undefined
      : component === "layer" || component === "attention" || component === "residual"
        ? `layer-${layer}`
        : items.find((item) => item.component === component)?.key;
  function center(key: string, focus = false) {
    const container = viewport.current;
    const node = container?.querySelector<HTMLButtonElement>(
      `[data-node="${key}"]`,
    );
    if (!container || !node) return;
    if (focus) node.focus({ preventScroll: true });
    container.scrollTo({
      left:
        container.scrollLeft +
        node.getBoundingClientRect().left -
        container.getBoundingClientRect().left -
        (container.clientWidth - node.clientWidth) / 2,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
    });
  }
  useEffect(() => {
    if (activeKey) center(activeKey);
  }, [activeKey]);
  useEffect(() => {
    const container = viewport.current;
    if (!container) return;
    const wheel = (event: WheelEvent) => {
      if (
        event.ctrlKey ||
        event.shiftKey ||
        Math.abs(event.deltaY) <= Math.abs(event.deltaX)
      )
        return;
      const next = Math.max(
        0,
        Math.min(
          container.scrollWidth - container.clientWidth,
          container.scrollLeft + event.deltaY,
        ),
      );
      if (next !== container.scrollLeft) {
        event.preventDefault();
        container.scrollLeft = next;
      }
    };
    container.addEventListener("wheel", wheel, { passive: false });
    return () => container.removeEventListener("wheel", wheel);
  }, []);
  function choose(item: JourneyItem) {
    if (item.layer != null) selectLayer(item.layer);
    else selectComponent(item.component);
    center(item.key);
  }
  function keyboard(event: KeyboardEvent<HTMLDivElement>) {
    const node =
      event.target instanceof HTMLElement
        ? event.target.closest<HTMLButtonElement>("[data-node]")
        : null;
    if (!node) return;
    const index = items.findIndex((item) => item.key === node.dataset.node);
    const next =
      event.key === "ArrowRight"
        ? Math.min(items.length - 1, index + 1)
        : event.key === "ArrowLeft"
          ? Math.max(0, index - 1)
          : event.key === "Home"
            ? 0
            : event.key === "End"
              ? items.length - 1
              : null;
    if (next === null) return;
    event.preventDefault();
    choose(items[next]);
    center(items[next].key, true);
  }
  function beginDrag(event: PointerEvent<HTMLDivElement>) {
    if (
      event.pointerType !== "mouse" ||
      event.button !== 0 ||
      (event.target instanceof HTMLElement && event.target.closest("button"))
    )
      return;
    drag.current = { x: event.clientX, left: event.currentTarget.scrollLeft };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  return (
    <div
      className={`model-journey ${mode} ${component === "attention" ? "attention-zoom" : ""}`}
    >
      <div className="journey-heading">
        <div>
          <p className="eyebrow">THE COMPUTATION</p>
          <h2>Model Journey</h2>
        </div>
        <div className="journey-tools">
          {overview && (
            <label className="layer-jump">
              Jump to layer{" "}
              <select
                aria-label="Jump to layer"
                value={layer ?? ""}
                onChange={(event) => {
                  const index = Number(event.target.value);
                  selectLayer(index);
                  center(`layer-${index}`);
                }}
              >
                <option value="" disabled>
                  Select
                </option>
                {indices.map((index) => (
                  <option key={index} value={index}>
                    L{index}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button
            className="secondary-button"
            onClick={() => {
              setMode("compact");
              viewport.current?.scrollTo({ left: 0, behavior: "smooth" });
            }}
          >
            Fit Journey
          </button>
          <button
            className="text-button"
            onClick={() => {
              setMode("comfortable");
              viewport.current?.scrollTo({ left: 0, behavior: "smooth" });
            }}
          >
            Reset
          </button>
        </div>
      </div>
      <nav className="journey-map" aria-label="Journey overview">
        {start.map((item) => (
          <button key={item.key} disabled={!id} onClick={() => choose(item)}>
            {item.label}
          </button>
        ))}
        <button
          className="map-transformer"
          disabled={!id}
          onClick={() => indices.length && choose(items[3])}
        >
          <span>Transformer</span>
          <small>
            {overview ? `${indices.length} layers` : "Decoder stack"}
          </small>
        </button>
        {tail.map((item) => (
          <button
            key={item.key}
            className={item.component === "prediction" ? "map-prediction" : ""}
            disabled={!id}
            onClick={() => choose(item)}
          >
            <span>{item.label}</span>
            {item.detail && (
              <small>
                {item.detail}{" "}
                <span className="map-probability-label">probability</span>
              </small>
            )}
          </button>
        ))}
      </nav>
      {layers.error && (
        <ApiErrorNotice
          error={layers.error}
          onRetry={() => void layers.refetch()}
        />
      )}
      <div
        className="journey-viewport"
        ref={viewport}
        role="region"
        aria-label="Horizontal model computation"
        tabIndex={0}
        onKeyDown={keyboard}
        onPointerDown={beginDrag}
        onPointerMove={(event) => {
          if (drag.current)
            event.currentTarget.scrollLeft =
              drag.current.left - (event.clientX - drag.current.x);
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        <div className="journey-rail">
          <div className="rail-section input-section">
            <span className="region-label">INPUT REPRESENTATION</span>
            <div className="rail-nodes">
              {start.map((item) => (
                <JourneyNode
                  key={item.key}
                  item={item}
                  selected={activeKey === item.key}
                  enabled={!!id}
                  onSelect={() => choose(item)}
                />
              ))}
            </div>
          </div>
          <div className="rail-section transformer-region">
            <span className="region-label">
              <span>
                TRANSFORMER
                {overview
                  ? ` · ${layers.data?.total ?? indices.length} DECODER LAYERS`
                  : ""}
              </span>
            </span>
            <div className="rail-nodes">
              {indices.length ? (
                layerItems.map((item) => (
                  <JourneyNode
                    key={item.key}
                    item={item}
                    selected={activeKey === item.key}
                    enabled={!!id}
                    onSelect={() => choose(item)}
                  />
                ))
              ) : (
                <div className="untraced-stack">
                  {busy
                    ? "Capturing the real computation…"
                    : "Layer nodes appear from your trace"}
                </div>
              )}
            </div>
          </div>
          <div className="rail-section output-section">
            <span className="region-label">VOCABULARY PROJECTION</span>
            <div className="rail-nodes">
              {tail.map((item) => (
                <JourneyNode
                  key={item.key}
                  item={item}
                  selected={activeKey === item.key}
                  enabled={!!id}
                  onSelect={() => choose(item)}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="journey-caption">
        <p>Scroll horizontally · drag the rail · use ← / → on a node</p>
        <span>
          {token != null
            ? `Token ${token} stays selected`
            : mode === "compact"
              ? "Fit overview · readable labels preserved"
              : "Select any stage to inspect"}
        </span>
      </div>
    </div>
  );
}
