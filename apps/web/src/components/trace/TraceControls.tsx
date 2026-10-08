import type { FormEvent } from "react";
import type { ModelStatus, TraceOverview } from "@/lib/api/client";

export function TraceControls({
  prompt,
  setPrompt,
  model,
  busy,
  phase,
  overview,
  onRun,
  onLoad,
  apiUnavailable,
}: {
  prompt: string;
  setPrompt: (text: string) => void;
  model?: ModelStatus;
  busy: boolean;
  phase: string;
  overview?: TraceOverview | null;
  onRun: () => void;
  onLoad: () => void;
  apiUnavailable: boolean;
}) {
  function submit(event: FormEvent) {
    event.preventDefault();
    onRun();
  }
  const modelLabel = apiUnavailable
    ? "API offline"
    : model?.state === "ready"
      ? "Model ready"
      : model?.state === "loading"
        ? "Loading model"
        : model?.state === "failed"
          ? "Model failed"
          : model
            ? "Model unloaded"
            : "Connecting to API";
  return (
    <header className="control-header">
      <div className="brand-row">
        <div className="brand">
          <span className="brand-symbol" aria-hidden="true">
            λ
          </span>
          <div>
            <p className="eyebrow">INTERPRETABILITY WORKSPACE</p>
            <h1>LLM Visualization Lab</h1>
          </div>
        </div>
        <div className="model-identity">
          <strong>Qwen3-0.6B</strong>
          <span
            className={`model-state ${model?.state === "ready" && !apiUnavailable ? "is-ready" : ""}`}
          >
            <i aria-hidden="true" />
            {modelLabel}
          </span>
          {model && ["unloaded", "failed"].includes(model.state) && (
            <button className="text-button" onClick={onLoad}>
              Load model
            </button>
          )}
        </div>
      </div>
      <form className="prompt-form" onSubmit={submit}>
        <div className="prompt-field">
          <label htmlFor="prompt">Raw prompt</label>
          <textarea
            id="prompt"
            rows={1}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Enter text to trace"
            aria-describedby="prompt-context"
          />
        </div>
        <button
          className="trace-button"
          type="submit"
          disabled={busy || model?.state !== "ready" || apiUnavailable}
        >
          <span>{busy ? "TRACING MODEL…" : "TRACE MODEL"}</span>
          <span aria-hidden="true">↗</span>
        </button>
      </form>
      <div className="control-caption">
        <p id="prompt-context">
          {overview && prompt !== overview.input.text
            ? "Prompt changed · the journey still shows the previous trace. Run to update."
            : "Raw text · exact tokenizer input · one forward pass"}
        </p>
        <span className="trace-phase" role="status" aria-live="polite">
          <i className={busy ? "status-pulse" : ""} aria-hidden="true" />
          {phase}
        </span>
      </div>
    </header>
  );
}
