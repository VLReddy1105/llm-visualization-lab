import { useQuery } from "@tanstack/react-query";
import type { TraceOverview } from "@/lib/api/client";
import { tokensQuery } from "@/lib/api/queries";
import { useSelection } from "@/state/selection";
import { number, RawData } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { LayerInspector } from "./LayerInspector";
import { LogitsInspector } from "./LogitsInspector";
import { ComponentInspector } from "./ComponentInspector";

function TokenInspector({
  id,
  index,
  overview,
}: {
  id: string;
  index: number;
  overview: TraceOverview;
}) {
  const tokens = useQuery(tokensQuery(id));
  if (tokens.error) return <ApiErrorNotice error={tokens.error} />;
  const token = tokens.data?.find((item) => item.token_index === index);
  if (!token) return <p role="status">Loading token metadata…</p>;
  return (
    <div className="token-inspector">
      <div className="token-display">
        <p className="eyebrow">EXACT DECODED TOKEN</p>
        <strong>{JSON.stringify(token.decoded_token)}</strong>
        <p>This position remains selected as you explore the model.</p>
      </div>
      <dl className="facts">
        <div>
          <dt>Native tokenizer piece</dt>
          <dd>
            <code>{token.tokenizer_piece}</code>
          </dd>
        </div>
        <div>
          <dt>Token ID</dt>
          <dd>{token.token_id}</dd>
        </div>
        <div>
          <dt>Input index / position</dt>
          <dd>
            {token.token_index} / {token.position_id}
          </dd>
        </div>
        <div>
          <dt>Character span</dt>
          <dd>
            {token.character_start != null
              ? `${token.character_start}:${token.character_end}`
              : "Not available"}
          </dd>
        </div>
        <div>
          <dt>Capture coverage</dt>
          <dd>
            {overview.capture.token_indices.includes(index)
              ? "Retained token position"
              : "Not captured in this trace"}
          </dd>
        </div>
        <div>
          <dt>Captured layers</dt>
          <dd>{overview.available.layers.length} layer summaries</dd>
        </div>
      </dl>
      <RawData label="Raw token metadata" data={token} />
    </div>
  );
}

const names: Record<string, string> = {
  input: "Input",
  tokens: "Tokenization",
  "embedding.output": "Embedding",
  "output.normalized": "Final Norm",
  "output.lm_head_input": "LM Head",
  logits: "Logits",
  prediction: "Next-token prediction",
  token: "Token",
};
export function Inspector({
  id,
  overview,
}: {
  id: string | null;
  overview?: TraceOverview | null;
}) {
  const component = useSelection((state) => state.selectedComponent);
  const layer = useSelection((state) => state.selectedLayer);
  const token = useSelection((state) => state.selectedToken);
  const select = useSelection((state) => state.selectComponent);
  const title =
    component === "layer"
      ? `Layer ${layer}`
      : component === "token"
        ? `Token ${token}`
        : names[component];
  return (
    <section className="inspector-panel" aria-labelledby="inspector-heading">
      <div className="inspector-header">
        <div>
          <p className="eyebrow">CONTEXT INSPECTOR</p>
          <h3 id="inspector-heading">
            {id ? title : "Look inside the computation"}
          </h3>
        </div>
        {id && (
          <nav
            className="inspector-breadcrumb"
            aria-label="Inspector breadcrumb"
          >
            <button className="text-button" onClick={() => select("input")}>
              Model Journey
            </button>
            <span aria-hidden="true">/</span>
            <span>{title}</span>
            {token != null && component !== "token" && (
              <span className="selected-token-context">
                Token {token} selected
              </span>
            )}
          </nav>
        )}
      </div>
      <div className="inspector-body">
        {!id || !overview ? (
          <p className="inspector-empty">
            Run a trace, then select a token, layer, or output stage to inspect
            its real data.
          </p>
        ) : component === "layer" && layer != null ? (
          <LayerInspector id={id} index={layer} overview={overview} />
        ) : component === "token" && token != null ? (
          <TokenInspector id={id} index={token} overview={overview} />
        ) : component === "logits" || component === "prediction" ? (
          <LogitsInspector id={id} />
        ) : [
            "embedding.output",
            "output.normalized",
            "output.lm_head_input",
          ].includes(component) ? (
          <ComponentInspector
            id={id}
            component={component}
            overview={overview}
          />
        ) : component === "tokens" ? (
          <div className="input-inspector">
            <div>
              <p>
                {overview.input.sequence_length} exact tokens from the pinned
                tokenizer. Select a token in the strip above.
              </p>
              <p className="fine-print">
                Decoded strings preserve spaces; native pieces retain tokenizer
                markers. Individual byte-fragment tokens may not decode to the
                original text in isolation.
              </p>
            </div>
            <dl className="facts">
              <div>
                <dt>Tokenizer revision</dt>
                <dd>{overview.model.tokenizer_revision}</dd>
              </div>
              <div>
                <dt>Special-token insertion</dt>
                <dd>Disabled</dd>
              </div>
              <div>
                <dt>Input mode</dt>
                <dd>{overview.input.mode}</dd>
              </div>
            </dl>
          </div>
        ) : (
          <div className="input-inspector">
            <div>
              <p className="raw-prompt">{overview.input.text}</p>
              <p className="fine-print">
                Exact raw input · no chat template or inserted special tokens
              </p>
            </div>
            <dl className="facts">
              <div>
                <dt>Characters / tokens</dt>
                <dd>
                  {Array.from(overview.input.text).length} /{" "}
                  {overview.input.sequence_length}
                </dd>
              </div>
              <div>
                <dt>Execution</dt>
                <dd>
                  {overview.model.device} · {overview.model.dtype}
                </dd>
              </div>
              <div>
                <dt>Trace duration</dt>
                <dd>{number(overview.summary.duration_ms)} ms</dd>
              </div>
              <div>
                <dt>Capture</dt>
                <dd>{overview.capture.profile}</dd>
              </div>
            </dl>
          </div>
        )}
      </div>
    </section>
  );
}
