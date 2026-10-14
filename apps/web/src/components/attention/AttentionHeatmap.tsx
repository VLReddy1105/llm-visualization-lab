import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import type { TokenMetadata } from "@/lib/api/client";
import { useSelection } from "@/state/selection";
import { visibleWhitespace } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import {
  attentionComponent,
  exact,
  isMasked,
  sliceQuery,
  tokenDescription,
} from "./data";

export const MATRIX_WINDOW = 16;
export function AttentionHeatmap({
  id,
  layer,
  head,
  query,
  selectedKey,
  tokens,
}: {
  id: string;
  layer: number;
  head: number;
  query: number;
  selectedKey: number;
  tokens: TokenMetadata[];
}) {
  const select = useSelection((s) => s.selectRelation);
  const [tileWindow, setWindow] = useState<{
    q: number;
    k: number;
    row: number;
    column: number;
  } | null>(null);
  const row =
    tileWindow?.q === query && tileWindow.k === selectedKey
      ? tileWindow.row
      : Math.floor(query / MATRIX_WINDOW) * MATRIX_WINDOW;
  const column =
    tileWindow?.q === query && tileWindow.k === selectedKey
      ? tileWindow.column
      : Math.floor(selectedKey / MATRIX_WINDOW) * MATRIX_WINDOW;
  const rows = tokens.slice(row, row + MATRIX_WINDOW);
  const columns = tokens.slice(column, column + MATRIX_WINDOW);
  const selectors = {
    batch: { index: 0 },
    query_token: { start: row, stop: row + rows.length, step: 1 },
    key_token: { start: column, stop: column + columns.length, step: 1 },
  };
  const weights = useQuery(
    sliceQuery(id, attentionComponent(layer, "weights_reconstructed"), {
      ...selectors,
      query_head: { index: head },
    }),
  );
  const mask = useQuery(
    sliceQuery(id, "position.attention_mask", {
      ...selectors,
      mask_head: { index: 0 },
    }),
  );
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const scroller = useRef<HTMLDivElement>(null);
  const [focus, setFocus] = useState<number | null>(null);
  const selectedIndex = (query - row) * columns.length + selectedKey - column;
  const selectedVisible =
    query >= row &&
    query < row + rows.length &&
    selectedKey >= column &&
    selectedKey < column + columns.length;
  const stop =
    focus != null && focus < rows.length * columns.length
      ? focus
      : selectedVisible
        ? selectedIndex
        : 0;
  useEffect(() => {
    if (!selectedVisible || !weights.data || !mask.data) return;
    const center = () => {
      const element = buttons.current[selectedIndex];
      const container = scroller.current;
      if (element && container) {
        const cellBounds = element.getBoundingClientRect();
        const containerBounds = container.getBoundingClientRect();
        container.scrollTo({
          left:
            container.scrollLeft +
            cellBounds.left -
            containerBounds.left -
            (container.clientWidth - cellBounds.width) / 2,
          top:
            container.scrollTop +
            cellBounds.top -
            containerBounds.top -
            (container.clientHeight - cellBounds.height) / 2,
          behavior: "instant",
        });
      }
    };
    center();
    window.addEventListener("resize", center);
    return () => window.removeEventListener("resize", center);
  }, [
    query,
    selectedKey,
    selectedIndex,
    selectedVisible,
    weights.data,
    mask.data,
    tileWindow,
  ]);
  const move = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next: number;
    if (event.key === "ArrowRight")
      next = Math.min(index + 1, rows.length * columns.length - 1);
    else if (event.key === "ArrowLeft") next = Math.max(0, index - 1);
    else if (event.key === "ArrowDown")
      next = Math.min(index + columns.length, rows.length * columns.length - 1);
    else if (event.key === "ArrowUp")
      next = Math.max(0, index - columns.length);
    else if (event.key === "Home")
      next = event.ctrlKey
        ? 0
        : Math.floor(index / columns.length) * columns.length;
    else if (event.key === "End")
      next = event.ctrlKey
        ? rows.length * columns.length - 1
        : Math.floor(index / columns.length) * columns.length +
          columns.length -
          1;
    else return;
    event.preventDefault();
    setFocus(next);
    buttons.current[next]?.focus();
  };
  const shift = (r: number, c: number) => {
    setWindow({ q: query, k: selectedKey, row: r, column: c });
    setFocus(0);
  };
  return (
    <section className="heatmap-section" aria-labelledby="heatmap-heading">
      <div className="section-heading">
        <div>
          <h5 id="heatmap-heading">Attention matrix</h5>
          <p className="fine-print">
            Query rows × key columns · full-row softmax
          </p>
        </div>
        <span className="evidence-label">RECONSTRUCTED</span>
      </div>
      {tokens.length > MATRIX_WINDOW && (
        <div className="matrix-window-controls">
          <span>
            Rows {row}–{row + rows.length - 1} · keys {column}–
            {column + columns.length - 1}
          </span>
          <button
            disabled={row === 0}
            onClick={() => shift(Math.max(0, row - MATRIX_WINDOW), column)}
          >
            Previous queries
          </button>
          <button
            disabled={row + MATRIX_WINDOW >= tokens.length}
            onClick={() => shift(row + MATRIX_WINDOW, column)}
          >
            Next queries
          </button>
          <button
            disabled={column === 0}
            onClick={() => shift(row, Math.max(0, column - MATRIX_WINDOW))}
          >
            Previous keys
          </button>
          <button
            disabled={column + MATRIX_WINDOW >= tokens.length}
            onClick={() => shift(row, column + MATRIX_WINDOW)}
          >
            Next keys
          </button>
          <button
            onClick={() =>
              shift(
                Math.floor(query / MATRIX_WINDOW) * MATRIX_WINDOW,
                Math.floor(selectedKey / MATRIX_WINDOW) * MATRIX_WINDOW,
              )
            }
          >
            Center selection
          </button>
        </div>
      )}
      {weights.error ? (
        <ApiErrorNotice
          error={weights.error}
          onRetry={() => void weights.refetch()}
        />
      ) : mask.error ? (
        <ApiErrorNotice
          error={mask.error}
          onRetry={() => void mask.refetch()}
        />
      ) : !weights.data || !mask.data ? (
        <p role="status">Loading bounded attention matrix…</p>
      ) : (
        <div className="heatmap-scroll" ref={scroller}>
          <table
            className="attention-matrix"
            role="grid"
            aria-label={`Layer ${layer}, query head ${head} reconstructed attention weights`}
            aria-rowcount={tokens.length + 1}
            aria-colcount={tokens.length + 1}
          >
            <thead>
              <tr role="row">
                <th role="columnheader" className="matrix-corner">
                  Q ↓ / K →
                </th>
                {columns.map((t) => (
                  <th
                    role="columnheader"
                    key={t.token_index}
                    title={tokenDescription(t)}
                  >
                    <small>{t.token_index}</small>
                    {visibleWhitespace(t.decoded_token)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((q, r) => (
                <tr
                  role="row"
                  key={q.token_index}
                  aria-rowindex={q.token_index + 2}
                >
                  <th role="rowheader" title={tokenDescription(q)}>
                    <small>{q.token_index}</small>
                    {visibleWhitespace(q.decoded_token)}
                  </th>
                  {columns.map((k, c) => {
                    const i = r * columns.length + c;
                    const weight = weights.data.values[i];
                    const masked = isMasked(mask.data.values[i]);
                    const selected =
                      q.token_index === query && k.token_index === selectedKey;
                    const label = `Query ${q.token_index} ${JSON.stringify(q.decoded_token)}, key ${k.token_index} ${JSON.stringify(k.decoded_token)}; ${masked ? "causally masked; unavailable relation" : `attention weight ${exact(weight)}`}; layer ${layer}, query head ${head}. ${tokenDescription(q)}. ${tokenDescription(k)}`;
                    return (
                      <td
                        role="gridcell"
                        aria-colindex={k.token_index + 2}
                        aria-selected={selected}
                        key={k.token_index}
                      >
                        <button
                          ref={(element) => {
                            buttons.current[i] = element;
                          }}
                          tabIndex={i === stop ? 0 : -1}
                          onFocus={() => setFocus(i)}
                          aria-label={label}
                          title={label}
                          aria-disabled={masked}
                          data-query={q.token_index}
                          data-key={k.token_index}
                          data-weight={exact(weight)}
                          data-masked={masked}
                          className={`matrix-cell ${masked ? "masked-cell" : ""} ${selected ? "selected-cell" : ""}`}
                          style={
                            masked
                              ? undefined
                              : {
                                  backgroundColor: `rgba(48, 106, 173, ${0.05 + (weight ?? 0) * 0.7})`,
                                  color: "#162433",
                                }
                          }
                          onKeyDown={(e) => move(e, i)}
                          onClick={() => {
                            if (!masked) select(q.token_index, k.token_index);
                          }}
                        >
                          {masked ? (
                            <span aria-hidden="true">╱</span>
                          ) : weight == null ? (
                            "—"
                          ) : (
                            `${(weight * 100).toFixed(1)}%`
                          )}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="matrix-legend">
        <span>
          <i className="legend-low" />0
        </span>
        <span className="legend-ramp" aria-hidden="true" />
        <span>1</span>
        <span>
          <i className="masked-cell" />
          Causally masked
        </span>
      </div>
      <p className="fine-print">
        Arrow keys move focus; Enter selects an allowed relation. Masked cells
        use the retained native mask. A bounded window preserves the original
        weights.
      </p>
    </section>
  );
}
