import { useSelection } from "@/state/selection";
import { useState } from "react";
import type { SamplingParameters, TraceOverview } from "@/lib/api/client";
import { number, RawData } from "@/components/ui/ScientificData";
import { useAnalysis } from "./useAnalysis";
import { AnalysisStatus } from "./AnalysisStatus";

export function SamplingLab({
  id,
  source,
}: {
  id: string;
  source: TraceOverview;
}) {
  const defaults: SamplingParameters = {
    token_index: source.input.sequence_length - 1,
    temperature: 1,
    top_k: 0,
    top_p: 1,
    display_top_k: 10,
    mode: "sample",
  };
  const settings = useSelection((s) => s.samplingSettings);
  const setParameters = useSelection((s) => s.setSamplingSettings);
  const parameters = { ...settings, token_index: defaults.token_index };
  const [draft, setDraft] = useState(parameters);
  const analysis = useAnalysis(id, { kind: "sampling_preview", parameters });
  const data =
    analysis.ready && analysis.data?.result?.kind === "sampling_preview"
      ? analysis.data.result
      : null;
  return (
    <div className="sampling-lab">
      <p>
        Preview the distribution from{" "}
        <strong>saved final-position logits</strong>, at input position{" "}
        {defaults.token_index}, for next position {defaults.token_index + 1}.
        This preview draws no token, advances no position and runs no
        transformer forward.
      </p>
      <form
        className="sampling-controls"
        onSubmit={(e) => {
          e.preventDefault();
          setParameters({
            temperature: draft.temperature,
            top_k: draft.top_k,
            top_p: draft.top_p,
            display_top_k: draft.display_top_k,
            mode: draft.mode,
          });
        }}
      >
        <label>
          Temperature
          <input
            type="number"
            min="0.000001"
            max="100"
            step="any"
            required
            value={Number.isNaN(draft.temperature) ? "" : draft.temperature}
            onChange={(e) =>
              setDraft({ ...draft, temperature: e.target.valueAsNumber })
            }
          />
        </label>
        <label>
          Top-k (0 disables)
          <input
            type="number"
            min="0"
            max={source.model.architecture.vocabulary_size}
            step="1"
            required
            value={Number.isNaN(draft.top_k) ? "" : draft.top_k}
            onChange={(e) =>
              setDraft({ ...draft, top_k: e.target.valueAsNumber })
            }
          />
        </label>
        <label>
          Top-p
          <input
            type="number"
            min="0.000001"
            max="1"
            step="any"
            required
            value={Number.isNaN(draft.top_p) ? "" : draft.top_p}
            onChange={(e) =>
              setDraft({ ...draft, top_p: e.target.valueAsNumber })
            }
          />
        </label>
        <label>
          Displayed candidates
          <input
            type="number"
            min="1"
            max="100"
            step="1"
            required
            value={Number.isNaN(draft.display_top_k) ? "" : draft.display_top_k}
            onChange={(e) =>
              setDraft({ ...draft, display_top_k: e.target.valueAsNumber })
            }
          />
        </label>
        <button className="primary-button" type="submit">
          Apply preview
        </button>
        <button
          className="secondary-button"
          type="button"
          onClick={() => {
            setDraft(defaults);
            setParameters({
              temperature: 1,
              top_k: 0,
              top_p: 1,
              display_top_k: 10,
              mode: "sample",
            });
          }}
        >
          Reset defaults
        </button>
      </form>
      <AnalysisStatus analysis={analysis} />
      {data && (
        <>
          <p className="sampling-applied" role="status">
            Applied: temperature {data.parameters.temperature} · top-k{" "}
            {data.parameters.top_k} · top-p {data.parameters.top_p}.{" "}
            {data.eligible_count.toLocaleString()} eligible vocabulary entries.
          </p>
          <p className="fine-print">
            Adjusted sum {data.probability_sum.toPrecision(10)} · omitted
            adjusted mass {number(data.omitted_probability_mass, 7)}. Display
            count does not change the distribution.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Token / native piece</th>
                  <th>ID</th>
                  <th>Raw logit</th>
                  <th>Processed logit</th>
                  <th>Model probability</th>
                  <th>Sampling-adjusted probability</th>
                </tr>
              </thead>
              <tbody>
                {data.candidates.map((c) => (
                  <tr key={c.token_id}>
                    <td>{c.rank}</td>
                    <td>
                      <strong>{JSON.stringify(c.decoded_token)}</strong>
                      <small>{c.tokenizer_piece}</small>
                    </td>
                    <td>{c.token_id}</td>
                    <td title={String(c.raw_logit)}>
                      {number(c.raw_logit, 6)}
                    </td>
                    <td title={String(c.processed_logit)}>
                      {number(c.processed_logit, 6)}
                    </td>
                    <td title={String(c.model_probability)}>
                      {number(c.model_probability, 7)}
                    </td>
                    <td title={String(c.probability)}>
                      <strong>{number(c.probability, 7)}</strong>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="science-caveat">
            This is a sampling distribution preview, not a generated
            continuation. Model probabilities and sampling-adjusted
            probabilities use different settings.
          </p>
          <details>
            <summary>Sampling recipe and source</summary>
            <p>{data.processor_order.join(" → ")}</p>
            <p>{data.method}</p>
            <p>{data.rank_policy}</p>
            <p>{data.probability_policy}</p>
            <p>
              Trace {id} · {data.source.device} / {data.source.dtype}
            </p>
            <RawData label="Exact sampling preview" data={data} />
          </details>
        </>
      )}
    </div>
  );
}
