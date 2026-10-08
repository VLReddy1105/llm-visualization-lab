import { useQuery } from "@tanstack/react-query";
import { logitsQuery } from "@/lib/api/queries";
import { useSelection } from "@/state/selection";
import {
  number,
  percent,
  RawData,
  visibleWhitespace,
} from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";

export function LogitsInspector({ id }: { id: string }) {
  const query = useQuery(logitsQuery(id));
  const selected = useSelection((state) => state.selectedCandidateToken);
  const selectedToken = useSelection((state) => state.selectedToken);
  const select = useSelection((state) => state.selectCandidate);
  if (query.isPending)
    return <p role="status">Loading retained next-token logits…</p>;
  if (query.error)
    return (
      <ApiErrorNotice
        error={query.error}
        onRetry={() => void query.refetch()}
      />
    );
  const data = query.data;
  const first = data.top_k[0];
  const candidate = data.top_k.find((item) => item.token_id === selected);
  return (
    <div className="logits-inspector">
      <div className="prediction-summary">
        <p className="eyebrow">TOP NEXT TOKEN</p>
        {first && (
          <>
            <strong>{visibleWhitespace(first.decoded_token)}</strong>
            <span>{percent(first.probability)}</span>
          </>
        )}
        <p>
          Position {data.prediction_position} predicts position{" "}
          {data.next_position}.
        </p>
        <p className="fine-print">
          Full vocabulary · {number(data.vocabulary_size)} tokens
          <br />
          Argmax preview · no token has been generated.
        </p>
        {selectedToken != null &&
          selectedToken !== data.prediction_position && (
            <p className="fine-print">
              Input token {selectedToken} stays selected. These logits belong to
              the final prompt position.
            </p>
          )}
      </div>
      <div className="prediction-table">
        <table>
          <caption>
            Real next-token candidates · full-vocabulary probabilities
          </caption>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Token</th>
              <th>Probability</th>
              <th>Raw logit</th>
            </tr>
          </thead>
          <tbody>
            {data.top_k.map((item) => (
              <tr
                className={
                  selected === item.token_id ? "candidate-selected" : ""
                }
                key={item.token_id}
              >
                <td>{item.rank}</td>
                <td>
                  <button
                    className="candidate-button"
                    aria-pressed={selected === item.token_id}
                    title={`ID ${item.token_id} · ${item.tokenizer_piece} · Raw logit ${item.raw_logit}`}
                    onClick={() => select(item.token_id)}
                  >
                    {visibleWhitespace(item.decoded_token)}
                  </button>
                </td>
                <td className="probability-cell">
                  <span
                    className="probability-bar"
                    style={{ width: percent(item.probability) }}
                    aria-hidden="true"
                  />
                  <span>{percent(item.probability)}</span>
                </td>
                <td>{item.raw_logit.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="fine-print">
          Outside the displayed candidates:{" "}
          {percent(data.omitted_probability_mass)}. Probabilities are never
          renormalized over this table.
        </p>
        {candidate && (
          <p className="candidate-detail">
            Selected candidate: {JSON.stringify(candidate.decoded_token)} ·
            native {candidate.tokenizer_piece} · vocabulary ID{" "}
            {candidate.token_id} · logit {candidate.raw_logit}
          </p>
        )}
        <RawData
          label="Logits descriptors and normalization"
          data={{
            logits: data.logits,
            probabilities: data.probabilities,
            normalization: data.normalization,
          }}
        />
      </div>
    </div>
  );
}
