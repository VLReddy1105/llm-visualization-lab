import { useQuery } from "@tanstack/react-query";
import type { TraceOverview } from "@/lib/api/client";
import { tokensQuery } from "@/lib/api/queries";
import { useSelection } from "@/state/selection";
import { visibleWhitespace } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";

export function TokenStrip({
  id,
  overview,
}: {
  id: string;
  overview: TraceOverview;
}) {
  const selected = useSelection((state) => state.selectedToken);
  const select = useSelection((state) => state.selectToken);
  const tokens = useQuery(tokensQuery(id));
  return (
    <section className="token-strip" aria-label="Input tokens">
      <div className="token-strip-label">
        <span className="eyebrow">TOKENIZED INPUT</span>
        <span>
          {overview.input.sequence_length} tokens{" "}
          <span className="whitespace-key">· means space</span>
        </span>
      </div>
      {tokens.isPending ? (
        <p role="status">Loading real tokens…</p>
      ) : tokens.error ? (
        <ApiErrorNotice
          error={tokens.error}
          onRetry={() => void tokens.refetch()}
        />
      ) : (
        <div className="token-list">
          {tokens.data.map((token) => (
            <button
              className={`token-chip ${selected === token.token_index ? "selected" : ""}`}
              key={token.token_index}
              aria-pressed={selected === token.token_index}
              aria-label={`Token ${token.token_index}: ${JSON.stringify(token.decoded_token)}`}
              title={`Decoded ${JSON.stringify(token.decoded_token)} · Native ${token.tokenizer_piece} · ID ${token.token_id} · Position ${token.position_id}`}
              onClick={() => select(token.token_index)}
            >
              <span className="token-position">{token.token_index}</span>
              <span>{visibleWhitespace(token.decoded_token)}</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
