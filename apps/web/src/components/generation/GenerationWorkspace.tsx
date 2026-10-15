"use client";

import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, api, type GenerationRequest, type GenerationStep } from "@/lib/api/client";

const terminal = new Set(["completed", "stopped", "cancelled", "failed"]);
const emptySteps: GenerationStep[] = [];
const piece = (value: string) =>
  value.replaceAll(" ", "·").replaceAll("\n", "↵").replaceAll("\t", "⇥");

export function GenerationWorkspace({
  prompt,
  onOpenTrace,
}: {
  prompt: string;
  onOpenTrace: (step: GenerationStep, view: "journey" | "attention" | "residual" | "mlp") => void;
}) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [maxTokens, setMaxTokens] = useState(4);
  const [mode, setMode] = useState<"greedy" | "sample">("greedy");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.95);
  const [seed, setSeed] = useState(7);
  const [capture, setCapture] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const client = useQueryClient();
  const queryKey = ["generation", sessionId] as const;
  const session = useQuery({
    queryKey,
    queryFn: ({ signal }) => api.generation(sessionId!, signal),
    enabled: !!sessionId,
    staleTime: 0,
    refetchInterval: (query) => {
      const saved = query.state.data;
      if (saved && terminal.has(saved.status)) {
        return saved.steps?.some((step) => step.source_trace_id) ? 5000 : false;
      }
      return 700;
    },
  });
  const start = useMutation({
    mutationFn: (body: GenerationRequest) => api.createGeneration(body),
    onSuccess: (result) => {
      setSessionId(result.generation_id);
      setSelectedIndex(null);
      setLocalError(null);
      client.setQueryData(["generation", result.generation_id], result);
    },
  });
  const cancel = useMutation({
    mutationFn: (id: string) => api.cancelGeneration(id),
    onSuccess: (result) =>
      client.setQueryData(["generation", result.generation_id], result),
  });
  const remove = useMutation({ mutationFn: api.deleteGeneration });
  const data = session.data;
  const steps = data?.steps ?? emptySteps;
  const selected = useMemo(
    () => steps.find((step) => step.step_index === selectedIndex) ?? steps.at(-1) ?? null,
    [steps, selectedIndex],
  );
  const pending = data ? !terminal.has(data.status) : false;
  const sourceAvailable = !!selected?.source_trace_id &&
    !selected.source_capture_coverage.current_availability;

  useEffect(() => {
    if (!data?.expires_at) return;
    const delay = new Date(data.expires_at).getTime() - Date.now();
    const timer = setTimeout(() => {
      if (sessionId) {
        void client.cancelQueries({ queryKey: ["generation", sessionId] });
        client.removeQueries({ queryKey: ["generation", sessionId] });
      }
      setSessionId(null);
      setSelectedIndex(null);
      setLocalError("This generation session expired. Start a new run to create fresh decision records.");
    }, Math.max(0, delay));
    return () => clearTimeout(timer);
  }, [data?.expires_at, sessionId, client]);

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!prompt.trim()) {
      setLocalError("Enter a prompt containing non-whitespace text.");
      return;
    }
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 16) {
      setLocalError("New-token limit must be an integer from 1 to 16.");
      return;
    }
    if (mode === "sample") {
      if (!(temperature > 0 && temperature <= 100) || !Number.isFinite(temperature)) {
        setLocalError("Temperature must be greater than 0 and at most 100.");
        return;
      }
      if (!Number.isInteger(topK) || topK < 0 || topK > 151936) {
        setLocalError("Top-k must be zero (disabled) or no larger than the vocabulary.");
        return;
      }
      if (!(topP > 0 && topP <= 1) || !Number.isFinite(topP)) {
        setLocalError("Top-p must be greater than 0 and at most 1.");
        return;
      }
      if (!Number.isSafeInteger(seed) || seed < 0) {
        setLocalError("Seed must be a nonnegative safe integer.");
        return;
      }
    }
    setLocalError(null);
    start.reset();
    const body: GenerationRequest = {
      model_id: "qwen3-0.6b",
      input: { mode: "raw", text: prompt, add_special_tokens: false },
      max_new_tokens: maxTokens,
      sampling: {
        token_index: 0,
        mode,
        temperature: mode === "greedy" ? 1 : temperature,
        top_k: mode === "greedy" ? 0 : topK,
        top_p: mode === "greedy" ? 1 : topP,
        display_top_k: 10,
        ...(mode === "sample" ? { seed } : {}),
      },
      capture_profile: capture ? "deep" : "none",
      capture_layer: 12,
      retain_traces: capture ? Math.min(maxTokens, 4) : 0,
      use_cache: false,
      request_key: crypto.randomUUID(),
    };
    start.mutate(body);
  }

  function moveSelection(event: KeyboardEvent<HTMLDivElement>) {
    if (!steps.length) return;
    const current = selected?.step_index ?? 0;
    const next = event.key === "ArrowRight" ? Math.min(steps.length - 1, current + 1)
      : event.key === "ArrowLeft" ? Math.max(0, current - 1)
        : event.key === "Home" ? 0
          : event.key === "End" ? steps.length - 1 : null;
    if (next != null) {
      event.preventDefault();
      setSelectedIndex(next);
      (event.currentTarget.querySelector(`[data-step-index="${next}"]`) as HTMLElement | null)?.focus();
    }
  }

  const displayed = selected?.candidates
    .filter((candidate) => selected.sampling.mode === "greedy"
      ? candidate.raw_rank <= (selected.sampling.display_top_k ?? 10)
      : candidate.adjusted_rank != null && candidate.adjusted_rank <= (selected.sampling.display_top_k ?? 10))
    .sort((a, b) => selected.sampling.mode === "greedy"
      ? a.raw_rank - b.raw_rank
      : (a.adjusted_rank ?? Number.MAX_SAFE_INTEGER) - (b.adjusted_rank ?? Number.MAX_SAFE_INTEGER)) ?? [];
  const chartProbability = (candidate: (typeof displayed)[number]) =>
    selected?.sampling.mode === "greedy" ? candidate.model_probability : candidate.adjusted_probability ?? 0;
  const maxProbability = Math.max(...displayed.map(chartProbability), 1e-12);

  return (
    <section className="generation-workspace" aria-labelledby="generation-heading">
      <div className="generation-title">
        <div>
          <p className="eyebrow">STAGE 8 · REAL MODEL DECISIONS</p>
          <h2 id="generation-heading">Generation Journey</h2>
          <p>Each new token comes from one recorded full-prefix model forward.</p>
        </div>
        {data && <span className={`generation-status status-${data.status}`} role="status">{data.status} · {data.completed_steps}/{data.max_new_tokens} decisions</span>}
      </div>

      <form className="generation-controls" onSubmit={submit} noValidate>
        <label>New tokens
          <input aria-label="Max new tokens" type="number" min={1} max={16} step={1} value={maxTokens} onChange={(event) => setMaxTokens(event.target.valueAsNumber)} disabled={pending} />
        </label>
        <label>Mode
          <select aria-label="Generation mode" value={mode} onChange={(event) => setMode(event.target.value as "greedy" | "sample")} disabled={pending}>
            <option value="greedy">Greedy</option><option value="sample">Sampled</option>
          </select>
        </label>
        {mode === "sample" && <>
          <label>Temperature
            <input aria-label="Temperature" type="number" min="0.000001" max={100} step="any" value={temperature} onChange={(event) => setTemperature(event.target.valueAsNumber)} disabled={pending} />
          </label>
          <label>Top-k (0 disables)
            <input aria-label="Top-k" type="number" min={0} max={151936} step={1} value={topK} onChange={(event) => setTopK(event.target.valueAsNumber)} disabled={pending} />
          </label>
          <label>Top-p
            <input aria-label="Top-p" type="number" min="0.000001" max={1} step="any" value={topP} onChange={(event) => setTopP(event.target.valueAsNumber)} disabled={pending} />
          </label>
          <label>Seed
            <input aria-label="Sampling seed" type="number" min={0} max={Number.MAX_SAFE_INTEGER} step={1} value={seed} onChange={(event) => setSeed(event.target.valueAsNumber)} disabled={pending} />
          </label>
        </>}
        <label className="generation-capture"><input type="checkbox" checked={capture} onChange={(event) => setCapture(event.target.checked)} disabled={pending} /> Capture up to four original step traces</label>
        <div className="generation-actions">
          <button className="trace-button" type="submit" disabled={pending || start.isPending}>
            {start.isPending ? "STARTING…" : "START GENERATION"} <span aria-hidden="true">↗</span>
          </button>
          {pending && data && <button className="text-button" type="button" onClick={() => cancel.mutate(data.generation_id)} disabled={cancel.isPending}>Cancel after current forward</button>}
        </div>
      </form>
      <p className="generation-policy">Raw prompt · no chat template · cache disabled · {mode === "greedy" ? "sampling controls do not apply in greedy mode" : "Temperature → top-k → top-p → full-vocabulary sampling"}</p>
      {(localError || start.error || session.error) && <p className="generation-error" role="alert">{localError ?? (session.error instanceof ApiError && session.error.status === 410 ? "This generation has expired or was evicted; its science is no longer available." : (start.error ?? session.error)?.message)}</p>}

      {data && <>
        <div className="generation-output" aria-live="polite">
          <div><span className="eyebrow">ORIGINAL PROMPT</span><p>{data.original_prompt}</p></div>
          <span aria-hidden="true" className="generation-arrow">→</span>
          <div><span className="eyebrow">DECODED CONTINUATION</span><p>{data.decoded_continuation || (pending ? "Waiting for the first completed forward…" : "(empty continuation)")}</p></div>
          {!pending && <p className="generation-stop">Stop: {data.stop_reason ?? data.status}</p>}
        </div>
        <div className="generation-provenance">
          <span>Session <code>{data.generation_id}</code></span>
          <span>Model {data.model_id} · {data.model_revision.slice(0, 12)}</span>
          <span>{data.device} · {data.dtype} · cache off</span>
          <span>Forwards {data.attempted_forwards} attempted / {data.completed_steps} published</span>
          <span>Session JSON {data.result_bytes.toLocaleString()} B · reserved state {data.accounted_state_bytes.toLocaleString()} B</span>
        </div>
        <div className="generation-timeline-region">
          <p className="eyebrow">TOKEN-BY-TOKEN DECISIONS</p>
          {steps.length ? <div className="generation-timeline" role="listbox" aria-label="Generation steps" tabIndex={0} onKeyDown={moveSelection}>
            {steps.map((step) => <button key={step.step_id} role="option" aria-selected={selected?.step_index === step.step_index} data-step-index={step.step_index} className="generation-step-node" onClick={() => setSelectedIndex(step.step_index)}>
              <span className="generation-step-number">{step.step_index + 1}</span>
              <strong>{step.selected_display_text == null ? "Context changed" : JSON.stringify(step.selected_display_text)}</strong>
              <small>ID {step.selected_token_id} · position {step.selected_position}</small>
              <small>{step.forward_completion_status} · {step.forward_duration_ms.toFixed(1)} ms</small>
            </button>)}
            {pending && <span className="generation-waiting" role="status">Next forward waits for the current prefix</span>}
          </div> : <p className="generation-empty" role="status">{pending ? "Generation is queued; no token has been fabricated." : "No token decision completed."}</p>}
        </div>
      </>}

      {selected && <div className="generation-detail-grid">
        <section className="generation-card decision-card" aria-labelledby="decision-heading">
          <p className="eyebrow">SELECTED DECISION · STEP {selected.step_index + 1}</p>
          <h3 id="decision-heading">Selected token ID {selected.selected_token_id}</h3>
          <p className="generation-token-display">{selected.selected_display_text == null ? selected.display_note : JSON.stringify(selected.selected_display_text)}</p>
          <dl className="generation-facts">
            <div><dt>Exact prefix IDs</dt><dd><code>{selected.prefix_token_ids.join(" ")}</code></dd></div>
            <div><dt>Decision position</dt><dd>{selected.decision_position} · selected token is appended at {selected.selected_position}</dd></div>
            <div><dt>Raw argmax</dt><dd>ID {selected.raw_argmax_token_id}{selected.raw_argmax_token_id === selected.selected_token_id ? " (selected)" : " (different from selected)"}</dd></div>
            <div><dt>Selected raw logit</dt><dd>{selected.selected_raw_logit.toPrecision(7)}</dd></div>
            <div><dt>Model probability</dt><dd>{selected.selected_model_probability.toPrecision(7)}</dd></div>
            <div><dt>{selected.sampling.mode === "greedy" ? "Sampling-adjusted probability" : "Adjusted probability"}</dt><dd>{selected.selected_adjusted_probability == null ? "Not applicable in greedy mode" : selected.selected_adjusted_probability.toPrecision(7)}</dd></div>
            <div><dt>Eligible vocabulary</dt><dd>{Number(selected.distribution_statistics.eligible_count).toLocaleString()} / {selected.vocabulary_size.toLocaleString()}</dd></div>
            <div><dt>Forward execution</dt><dd><code>{selected.forward_execution_id}</code> · {selected.forward_duration_ms.toFixed(2)} ms</dd></div>
            <div><dt>Prefix digest</dt><dd><code>{selected.prefix_sha256}</code></dd></div>
          </dl>
        </section>
        <section className="generation-card candidate-card" aria-labelledby="candidate-heading">
          <p className="eyebrow">FULL-VOCABULARY DISTRIBUTION</p>
          <h3 id="candidate-heading">{selected.sampling.mode === "greedy" ? "Top raw model candidates" : "Top adjusted candidates"}</h3>
          <div className="generation-candidate-list" aria-label="Candidate probabilities">
            {displayed.map((candidate) => <div className="generation-candidate" key={candidate.token_id}>
              <div className="generation-candidate-label"><span>#{selected.sampling.mode === "greedy" ? candidate.raw_rank : candidate.adjusted_rank} · {JSON.stringify(piece(candidate.tokenizer_piece))} · ID {candidate.token_id}</span><span>{chartProbability(candidate).toPrecision(5)}</span></div>
              <div className="generation-probability-track"><span style={{ width: `${Math.max(0, chartProbability(candidate) / maxProbability * 100)}%` }} /></div>
              <small>Raw p {candidate.model_probability.toPrecision(5)} · {candidate.token_id === selected.raw_argmax_token_id ? "raw argmax" : candidate.token_id === selected.selected_token_id ? "selected" : ""}</small>
            </div>)}
          </div>
          <p className="generation-statline">Raw probability sum {Number(selected.distribution_statistics.raw_probability_sum).toFixed(8)} · {selected.sampling.mode === "greedy" ? "No sampling-adjusted distribution is applied" : `adjusted sum ${Number(selected.distribution_statistics.adjusted_probability_sum).toFixed(8)}`} · raw entropy {Number(selected.distribution_statistics.raw_entropy_nats).toFixed(4)} nats</p>
        </section>
        <section className="generation-card provenance-card" aria-labelledby="provenance-heading">
          <p className="eyebrow">SAMPLING AND SOURCE</p>
          <h3 id="provenance-heading">Decision provenance</h3>
          <dl className="generation-facts">
            <div><dt>Mode</dt><dd>{selected.sampling.mode}</dd></div>
            <div><dt>Temperature · top-k · top-p</dt><dd>{selected.sampling.temperature} · {selected.sampling.top_k} · {selected.sampling.top_p}</dd></div>
            <div><dt>Seed</dt><dd>{selected.sampling.seed ?? "Not used by greedy decoding"}</dd></div>
            <div><dt>RNG policy</dt><dd>{String(selected.provenance.rng_policy)}</dd></div>
            <div><dt>Processor order</dt><dd>{String(selected.provenance.sampling_policy)}</dd></div>
            <div><dt>Decoded prefix</dt><dd>{JSON.stringify(selected.decoded_prefix)}</dd></div>
            <div><dt>Full sequence after choice</dt><dd>{JSON.stringify(selected.decoded_full_sequence)}</dd></div>
            <div><dt>Original source trace</dt><dd>{selected.source_trace_id ? <><code>{selected.source_trace_id}</code>{selected.source_capture_coverage.current_availability ? ` · ${String(selected.source_capture_coverage.current_availability)}` : ""}</> : "Unavailable for this step"}</dd></div>
            <div><dt>Capture coverage</dt><dd>{selected.source_trace_id ? `${selected.source_capture_coverage.profile} · layer ${Array.isArray(selected.source_capture_coverage.layers) ? selected.source_capture_coverage.layers.join(", ") : "overview"} · token ${selected.decision_position}` : selected.capture_unavailable_reason}</dd></div>
          </dl>
          {sourceAvailable ? <div className="generation-inspector-actions">
            <button className="text-button" onClick={() => onOpenTrace(selected, "journey")}>Open original Model Journey</button>
            <button className="text-button" onClick={() => onOpenTrace(selected, "attention")}>Attention Microscope</button>
            <button className="text-button" onClick={() => onOpenTrace(selected, "residual")}>Residual Stream</button>
            <button className="text-button" onClick={() => onOpenTrace(selected, "mlp")}>MLP Explorer</button>
          </div> : <p className="science-caveat">{selected.source_trace_id ? "The original step trace expired, was deleted, or was evicted. Its decision summary remains, and historical selection never starts another forward." : `This decision summary is retained, but no detailed step trace was captured (${selected.capture_unavailable_reason}). Historical selection never starts another forward.`}</p>}
        </section>
      </div>}
      {data && !pending && <div className="generation-footer-actions"><button className="text-button" onClick={() => { remove.mutate(data.generation_id); client.removeQueries({ queryKey: ["generation", data.generation_id] }); setSessionId(null); setSelectedIndex(null); }}>Delete generation data</button></div>}
    </section>
  );
}
