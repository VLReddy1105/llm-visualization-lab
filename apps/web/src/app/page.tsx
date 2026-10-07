"use client";

import { useEffect, useState, type FormEvent } from "react";
import { api, apiBaseUrl, type ModelStatus, type NextTokenResponse } from "@/lib/api/client";

const stateLabels = { unloaded: "Not loaded", loading: "Loading", ready: "Ready", failed: "Error" };
const visibleToken = (value: string) => JSON.stringify(value);

export default function Home() {
  const [prompt, setPrompt] = useState("The capital of Germany is");
  const [topK, setTopK] = useState(10);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<NextTokenResponse | null>(null);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const [, status] = await Promise.all([api.health(), api.modelStatus()]);
        if (active) { setModel(status); setApiError(null); }
      } catch (error) {
        if (active) { setApiError(error instanceof Error ? error.message : "API unavailable."); setModel(null); }
      } finally {
        if (active) timer = setTimeout(refresh, 2000);
      }
    }
    void refresh();
    return () => { active = false; clearTimeout(timer); };
  }, []);

  async function loadModel() {
    setRunError(null);
    try { setModel(await api.loadModel()); }
    catch (error) { setRunError(error instanceof Error ? error.message : "Model loading failed."); }
  }

  async function run(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRunError(null);
    setResult(null);
    if (!prompt.trim()) { setRunError("Enter a non-empty prompt."); return; }
    if (!Number.isInteger(topK) || topK < 1 || topK > 100) {
      setRunError("Top predictions must be an integer between 1 and 100."); return;
    }
    setRunning(true);
    try { setResult(await api.nextToken(prompt, topK)); }
    catch (error) { setRunError(error instanceof Error ? error.message : "Inference failed."); }
    finally { setRunning(false); }
  }

  return (
    <main className="mx-auto max-w-5xl px-5 py-12 sm:px-8">
      <header className="mb-9">
        <p className="mb-3 text-xs font-semibold tracking-[0.18em] text-teal-700">STAGE 1 · REAL MODEL FOUNDATION</p>
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">LLM Visualization Lab</h1>
        <p className="mt-3 max-w-2xl text-base leading-7 text-slate-600">Inspect exact input tokens and the next-token distribution from a real Qwen forward pass.</p>
      </header>

      <section aria-label="Model and API status" className="mb-8 border-y border-slate-200 py-5">
        <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
          <div><span className="text-sm text-slate-500">Model</span><p className="font-medium">Qwen3-0.6B</p></div>
          <div aria-live="polite"><span className="text-sm text-slate-500">Model status</span><p className={model?.state === "ready" ? "font-medium text-teal-700" : "font-medium"}>{model ? stateLabels[model.state] : apiError ? "Unavailable" : "Connecting"}</p></div>
          <div><span className="text-sm text-slate-500">FastAPI</span><p className="font-medium">{apiError ? "Unavailable" : model ? "Connected" : "Connecting"}</p></div>
          {model?.device && <div><span className="text-sm text-slate-500">Execution</span><p className="font-mono text-sm">{model.device} · {model.dtype}</p></div>}
          {model && (model.state === "unloaded" || model.state === "failed") && <button type="button" onClick={loadModel} className="rounded border border-slate-300 px-4 py-2 text-sm">{model.state === "failed" ? "Retry model loading" : "Load model"}</button>}
        </div>
        {model?.loading_phase && <p className="mt-3 text-sm text-slate-600">{model.loading_phase.replaceAll("_", " ")}</p>}
        {model?.error && <p role="alert" className="mt-3 text-sm text-red-700">{model.error}</p>}
        <p className="mt-3 break-all font-mono text-xs text-slate-500">API: {apiBaseUrl}</p>
        {apiError && <p role="alert" className="mt-3 text-sm text-red-700">{apiError}</p>}
      </section>

      <form onSubmit={run} className="mb-10">
        <label htmlFor="prompt" className="mb-2 block font-medium">Prompt</label>
        <textarea id="prompt" value={prompt} onChange={event => setPrompt(event.target.value)} disabled={running} rows={3} maxLength={model?.max_prompt_characters ?? 8192} className="w-full resize-y rounded-lg border border-slate-300 bg-white p-4 text-base leading-7" />
        <div className="mt-4 flex flex-wrap items-center gap-4">
          <button type="submit" disabled={running || model?.state !== "ready" || !!apiError} className="rounded-lg bg-slate-900 px-6 py-3 font-medium text-white disabled:opacity-40">{running ? "Running forward pass…" : "Run next-token inference"}</button>
          <label htmlFor="top-k" className="flex items-center gap-2 text-sm text-slate-600">Top predictions<input id="top-k" type="number" min={1} max={100} step={1} value={topK} onChange={event => setTopK(Number(event.target.value))} disabled={running} className="w-20 rounded border border-slate-300 bg-white px-3 py-2 text-slate-900" /></label>
        </div>
        <p className="mt-3 text-sm text-slate-500">Raw text · no chat template · one next-token decision. Display count leaves the full-vocabulary probabilities unchanged.</p>
        {runError && <p role="alert" className="mt-4 text-sm text-red-700">{runError}</p>}
      </form>

      {result && <div aria-live="polite">
        <p className="mb-6 text-sm text-slate-600">{result.input.token_count} input tokens · {result.model.device} / {result.model.dtype} · {result.runtime.inference_duration_ms.toFixed(1)} ms total · {result.runtime.forward_duration_ms.toFixed(1)} ms forward</p>
        <section aria-labelledby="tokens-heading" className="mb-10">
          <h2 id="tokens-heading" className="mb-2 text-lg font-semibold">Input tokens</h2>
          <p className="mb-4 text-sm text-slate-500">Quoted decoded strings preserve visible whitespace; native pieces show tokenizer markers.</p>
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white"><table className="w-full text-sm"><caption className="sr-only">Exact input tokenization from Qwen&apos;s tokenizer</caption><thead><tr><th>Index / position</th><th>Decoded token</th><th>Native piece</th><th>Token ID</th></tr></thead><tbody>{result.input.tokens.map(token => <tr key={token.token_index}><td>{token.token_index} / {token.position_id}</td><td><code>{visibleToken(token.decoded_token)}</code></td><td><code>{token.tokenizer_piece}</code></td><td className="font-mono">{token.token_id}</td></tr>)}</tbody></table></div>
        </section>
        <section aria-labelledby="predictions-heading">
          <h2 id="predictions-heading" className="mb-2 text-lg font-semibold">Next token predictions</h2>
          <p className="mb-4 text-sm text-slate-500">Final input position {result.prediction.position} predicts position {result.prediction.next_position}. Probabilities use all {result.prediction.vocabulary_size.toLocaleString()} vocabulary logits.</p>
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white"><table className="w-full text-sm"><caption className="sr-only">Real next-token probabilities and raw logits</caption><thead><tr><th>Rank</th><th>Decoded token</th><th>Native piece / ID</th><th>Probability</th><th>Raw logit</th></tr></thead><tbody>{result.prediction.top_k.map(candidate => <tr key={candidate.token_id}><td>{candidate.rank}</td><td><code>{visibleToken(candidate.decoded_token)}</code></td><td><code>{candidate.tokenizer_piece}</code><span className="ml-2 text-xs text-slate-500">{candidate.token_id}</span></td><td className="font-mono">{(candidate.probability * 100).toFixed(6)}%</td><td className="font-mono">{candidate.raw_logit.toFixed(6)}</td></tr>)}</tbody></table></div>
          <p className="mt-3 text-sm text-slate-500">Probability mass outside this table: {(result.prediction.omitted_probability_mass * 100).toFixed(6)}%.</p>
        </section>
        <details className="mt-7 text-sm text-slate-600"><summary className="cursor-pointer">Execution provenance and exact input</summary><dl className="mt-3 space-y-2 break-all"><div><dt className="font-medium">Raw input</dt><dd><code>{visibleToken(result.input.text)}</code></dd></div><div><dt className="font-medium">Model revision</dt><dd className="font-mono">{result.model.revision}</dd></div><div><dt className="font-medium">Request ID</dt><dd className="font-mono">{result.request_id}</dd></div><div><dt className="font-medium">Runtime</dt><dd>Python {result.runtime.python_version} · PyTorch {result.runtime.torch_version} · Transformers {result.runtime.transformers_version} · {result.runtime.softmax_accumulation_dtype} output softmax / {result.runtime.probability_dtype} probabilities · eager attention · cache disabled · tracing disabled</dd></div></dl></details>
      </div>}
      <footer className="mt-12 border-t border-slate-200 pt-5 text-sm text-slate-500">Internal tracing and visualization begin in later stages.</footer>
    </main>
  );
}
