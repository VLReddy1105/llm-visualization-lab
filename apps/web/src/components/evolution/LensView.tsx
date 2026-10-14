import { useState } from "react";
import type { LensResult } from "@/lib/api/client";
import { number, RawData } from "@/components/ui/ScientificData";
import { useSelection } from "@/state/selection";
import { DepthChart } from "./DepthChart";

export function LensView({
  data,
  selected,
  onSelect,
  trajectories,
}: {
  data: LensResult;
  selected: number;
  onSelect: (index: number) => void;
  trajectories: boolean;
}) {
  const pins = useSelection((s) => s.pinnedCandidates);
  const pin = useSelection((s) => s.pinCandidate);
  const selectedCandidate = useSelection((s) => s.selectedCandidateToken);
  const selectCandidate = useSelection((s) => s.selectCandidate);
  const [enteredId, setEnteredId] = useState("");
  const [idError, setIdError] = useState("");
  const active = data.checkpoints[selected];
  const candidates = active.comparison_candidates;
  const rankId = candidates.some((c) => c.token_id === selectedCandidate)
    ? selectedCandidate!
    : candidates[0].token_id;
  const series = (field: "probability" | "raw_logit" | "rank") =>
    candidates
      .filter((c) => field !== "rank" || c.token_id === rankId)
      .map((candidate) => ({
        name: `${JSON.stringify(candidate.decoded_token)} #${candidate.token_id}`,
        values: data.checkpoints.map(
          (p) =>
            p.comparison_candidates.find(
              (c) => c.token_id === candidate.token_id,
            )?.[field] ?? null,
        ),
      }));
  const chart = { checkpoints: data.checkpoints, selected, onSelect };
  const check = data.final_logits_comparison;
  return (
    <>
      <div className="lens-method">
        <p>
          <strong>Logit Lens projections</strong> · raw residual → actual final
          RMSNorm → actual LM head. Captured final-normalized state uses the
          head only.
        </p>
        <p className="science-caveat">
          These diagnostic projections are not actual intermediate predictions,
          internal beliefs or causal explanations. Early layers may be unstable
          or misleading. Position {data.prediction_position} projects toward
          position {data.next_position}; it does not measure contribution to the
          final prompt position.
        </p>
      </div>
      <div
        className={`projection-check ${check?.passed === false || data.final_probabilities_comparison?.passed === false || !data.final_normalization_comparison.passed ? "check-failed" : ""}`}
        role="status"
      >
        {check ? (
          <>
            <strong>
              {check.passed && data.final_probabilities_comparison?.passed
                ? "Final projection agrees with retained logits"
                : "Final projection consistency FAILED"}
            </strong>
            <span>
              Maximum absolute logit error: {check.max_absolute_error} · atol{" "}
              {check.absolute_tolerance}, rtol {check.relative_tolerance}.
            </span>
          </>
        ) : (
          <strong>
            Original logits comparison unavailable at this position.
          </strong>
        )}
        <span>
          Final normalization check:{" "}
          {data.final_normalization_comparison.passed ? "PASS" : "FAIL"}.{" "}
          {data.comparison_note}
        </span>
      </div>
      <div className="candidate-controls">
        <label>
          Pin vocabulary ID{" "}
          <input
            type="number"
            min={0}
            max={data.source.vocabulary_size - 1}
            step={1}
            value={enteredId}
            onChange={(e) => setEnteredId(e.target.value)}
          />
        </label>
        <button
          className="secondary-button"
          onClick={() => {
            const value = Number(enteredId);
            if (
              !enteredId ||
              !Number.isInteger(value) ||
              value < 0 ||
              value >= data.source.vocabulary_size ||
              (pins.length >= 8 && !pins.includes(value))
            ) {
              setIdError(
                "Enter a valid vocabulary ID; at most eight pinned candidates.",
              );
              return;
            }
            setIdError("");
            pin(value);
            setEnteredId("");
          }}
        >
          Toggle pinned ID
        </button>
        {idError && <p role="alert">{idError}</p>}
      </div>
      <div
        className="candidate-chips"
        aria-label="Compared vocabulary candidates"
      >
        {candidates.map((c) => (
          <button
            key={c.token_id}
            className="secondary-button"
            aria-pressed={pins.includes(c.token_id)}
            onClick={() => pin(c.token_id)}
            disabled={!pins.includes(c.token_id) && pins.length >= 8}
            title={`${c.tokenizer_piece}; ID ${c.token_id}`}
          >
            {JSON.stringify(c.decoded_token)} <small>#{c.token_id}</small>
            {pins.includes(c.token_id) ? " · pinned ×" : " · pin"}
          </button>
        ))}
      </div>
      {trajectories && (
        <>
          <DepthChart
            {...chart}
            title="Candidate Probability Evolution"
            unit="Full-vocabulary probability"
            series={series("probability")}
          />
          <DepthChart
            {...chart}
            title="Candidate Logit Evolution"
            unit="Raw projected logit"
            series={series("raw_logit")}
          />
          <label className="rank-choice">
            Candidate for full-vocabulary rank{" "}
            <select
              value={rankId}
              onChange={(e) => selectCandidate(Number(e.target.value))}
            >
              {candidates.map((c) => (
                <option key={c.token_id} value={c.token_id}>
                  {JSON.stringify(c.decoded_token)} #{c.token_id}
                </option>
              ))}
            </select>
          </label>
          <DepthChart
            {...chart}
            title="Candidate Rank Evolution"
            unit="Rank 1 at top · full vocabulary"
            series={series("rank")}
            rank
          />
        </>
      )}
      <section
        className="lens-table"
        aria-label="Selected checkpoint projections"
      >
        <h4>{active.label} · top vocabulary projections</h4>
        <p className="fine-print">
          Full-vocabulary sum {active.probability_sum.toPrecision(10)} · entropy{" "}
          {number(active.entropy_nats, 5)} nats · omitted mass{" "}
          {number(active.omitted_probability_mass, 6)}.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Decoded token / native piece</th>
                <th>ID</th>
                <th>Logit</th>
                <th>Probability</th>
                <th>Compare</th>
              </tr>
            </thead>
            <tbody>
              {active.top_candidates.map((c) => (
                <tr key={c.token_id}>
                  <td>{c.rank}</td>
                  <td>
                    <strong>{JSON.stringify(c.decoded_token)}</strong>
                    <small>{c.tokenizer_piece}</small>
                  </td>
                  <td>{c.token_id}</td>
                  <td title={String(c.raw_logit)}>{number(c.raw_logit, 7)}</td>
                  <td title={String(c.probability)}>
                    {number(c.probability, 7)}
                  </td>
                  <td>
                    <button
                      className="text-button"
                      disabled={!pins.includes(c.token_id) && pins.length >= 8}
                      onClick={() => pin(c.token_id)}
                    >
                      {pins.includes(c.token_id) ? "Unpin" : "Pin"} #
                      {c.token_id}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <details className="layer-predictions">
        <summary>Layer Prediction Table · diagnostic projections</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Checkpoint</th>
                <th>Top candidates and full-vocabulary probabilities</th>
              </tr>
            </thead>
            <tbody>
              {data.checkpoints.map((c, i) => (
                <tr key={c.component_id}>
                  <td>
                    <button className="text-button" onClick={() => onSelect(i)}>
                      {c.label}
                    </button>
                  </td>
                  <td>
                    {c.top_candidates.map((t) => (
                      <span
                        className="inline-candidate"
                        key={t.token_id}
                        title={`ID ${t.token_id}; logit ${t.raw_logit}; rank ${t.rank}`}
                      >
                        {JSON.stringify(t.decoded_token)}{" "}
                        {number(t.probability, 5)}
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <p className="fine-print">
        Logits are unnormalized readout scores; probabilities include every
        vocabulary entry in their denominator; ranks describe relative order.
        Candidate defaults come from this position’s final readout. Pin an
        intermediate table entry or a vocabulary ID to compare it across depth.
      </p>
      <details>
        <summary>Readout provenance and exact values</summary>
        <p>{data.method}</p>
        <p>{data.rank_policy}</p>
        <p>{data.probability_policy}</p>
        <RawData label="Exact checkpoint projections" data={active} />
      </details>
    </>
  );
}
