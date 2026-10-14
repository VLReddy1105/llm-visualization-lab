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

export function TokenConnections({
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
  const selectors = { batch: { index: 0 }, query_token: { index: query } };
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
  const error = weights.error ?? mask.error;
  if (error)
    return (
      <ApiErrorNotice
        error={error}
        onRetry={() => {
          void weights.refetch();
          void mask.refetch();
        }}
      />
    );
  if (!weights.data || !mask.data)
    return <p role="status">Loading selected query distribution…</p>;
  const allowed = tokens.filter((_, i) => !isMasked(mask.data.values[i]));
  const width = Math.max(600, allowed.length * 116);
  return (
    <section
      className="token-connections"
      aria-labelledby="connections-heading"
    >
      <div className="section-heading">
        <h5 id="connections-heading">
          Attention distribution for selected query token
        </h5>
        <span className="fine-print">Line width = attention weight</span>
      </div>
      <div
        className="connection-scroll"
        tabIndex={0}
        aria-label="Allowed key connections, horizontally scrollable"
      >
        <div style={{ width }}>
          <svg
            viewBox={`0 0 ${width} 145`}
            role="img"
            aria-label={`Query ${JSON.stringify(tokens[query].decoded_token)} routes to ${allowed.length} allowed keys. Exact weights are on the key buttons below.`}
          >
            {allowed.map((t, i) => (
              <path
                key={t.token_index}
                d={`M ${width / 2} 37 C ${width / 2} 85, ${((i + 0.5) * width) / allowed.length} 68, ${((i + 0.5) * width) / allowed.length} 136`}
                fill="none"
                stroke={t.token_index === selectedKey ? "#2457b2" : "#8299b7"}
                strokeWidth={1 + (weights.data.values[t.token_index] ?? 0) * 12}
                opacity="0.85"
              >
                <title>
                  {tokenDescription(t)}; weight{" "}
                  {exact(weights.data.values[t.token_index])}
                </title>
              </path>
            ))}
            <rect
              x={width / 2 - 70}
              y="0"
              width="140"
              height="37"
              rx="8"
              fill="#edf3fb"
              stroke="#2457b2"
            />
            <text
              x={width / 2}
              y="24"
              textAnchor="middle"
              fill="#1d2938"
              fontSize="16"
            >
              Q{query} · {visibleWhitespace(tokens[query].decoded_token)}
            </text>
          </svg>
          <div className="connection-keys">
            {allowed.map((t) => (
              <button
                key={t.token_index}
                aria-pressed={selectedKey === t.token_index}
                title={`${tokenDescription(t)}; attention weight ${exact(weights.data.values[t.token_index])}`}
                aria-label={`Key ${t.token_index}: ${JSON.stringify(t.decoded_token)}, attention weight ${exact(weights.data.values[t.token_index])}`}
                onClick={() => select(query, t.token_index)}
              >
                <small>K{t.token_index}</small>
                <strong>{visibleWhitespace(t.decoded_token)}</strong>
                <span>
                  {weights.data.values[t.token_index] == null
                    ? "Unavailable"
                    : `${(weights.data.values[t.token_index]! * 100).toFixed(2)}%`}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
      <p className="fine-print">
        Hover or focus a key for its exact retained weight.
      </p>
    </section>
  );
}
