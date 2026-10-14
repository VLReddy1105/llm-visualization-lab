import { useQuery } from "@tanstack/react-query";
import { api, type TraceOverview } from "@/lib/api/client";
import { layerQuery, tokensQuery, traceKey } from "@/lib/api/queries";
import { useSelection } from "@/state/selection";
import { RawData, visibleWhitespace } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { LayerFlow } from "@/components/inspector/LayerFlow";
import { useAttentionCapture } from "./useAttentionCapture";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { TokenConnections } from "./TokenConnections";
import { RelationInspector } from "./RelationInspector";
import { tokenDescription } from "./data";

export function AttentionExplorer({
  sourceId,
  layer,
  source,
}: {
  sourceId: string;
  layer: number;
  source: TraceOverview;
}) {
  const capture = useAttentionCapture(sourceId, layer, source);
  const backLayer = useSelection((s) => s.selectLayer);
  const backModel = useSelection((s) => s.selectComponent);
  return (
    <div className="attention-explorer">
      <div className="zoom-navigation">
        <span>
          {source.model.hf_model_id} / Layer {layer} / Attention
        </span>
        <div>
          <button className="text-button" onClick={() => backLayer(layer)}>
            ← Back to Layer
          </button>
          <button className="text-button" onClick={() => backModel("input")}>
            Back to Model Journey
          </button>
        </div>
      </div>
      {!capture.ready ? (
        <>
          <LayerFlow
            index={layer}
            fallbackToken={source.input.sequence_length - 1}
            insideAttention
          />
          <div className="capture-attention">
            <p className="eyebrow">ATTENTION DETAIL</p>
            <h4>
              {capture.issue?.code === "TRACE_EXPIRED"
                ? "Detailed trace expired"
                : capture.issue
                  ? "Detailed trace unavailable"
                  : capture.pending
                    ? "Capturing layer attention…"
                    : "Detailed attention data was not captured in this trace."}
            </h4>
            <p>
              Capture the same raw prompt with Layer {layer} attention, Q/K/V
              and residual boundaries. This creates a separate model execution;
              the overview stays linked to its original trace.
            </p>
            {capture.id && (
              <p className="fine-print">
                Detailed trace: <code>{capture.id}</code>
              </p>
            )}
            {capture.issue && <ApiErrorNotice error={capture.issue} />}
            {capture.trace.error && !capture.issue && (
              <ApiErrorNotice
                error={capture.trace.error}
                onRetry={() => void capture.trace.refetch()}
              />
            )}
            {capture.capture.error && (
              <ApiErrorNotice error={capture.capture.error} />
            )}
            {capture.cancel.error && (
              <ApiErrorNotice error={capture.cancel.error} />
            )}
            {capture.trace.data?.status === "failed" && (
              <>
                <p role="alert">
                  Detailed capture failed. No tensors are available for this
                  execution.
                </p>
                <RawData
                  label="Capture failure details"
                  data={capture.trace.data.error}
                />
              </>
            )}
            {capture.timedOut && capture.pending && (
              <p role="alert">
                Capture is taking longer than two minutes. Check its status or
                cancel before trying again.
              </p>
            )}
            {capture.pending && !capture.issue ? (
              <div className="capture-actions">
                <button
                  className="secondary-button"
                  onClick={() => void capture.trace.refetch()}
                >
                  Check detailed trace
                </button>
                <button
                  className="secondary-button"
                  disabled={capture.cancel.isPending}
                  onClick={() => capture.cancel.mutate()}
                >
                  Cancel detailed capture
                </button>
              </div>
            ) : (
              <button
                className="primary-button"
                disabled={capture.capture.isPending}
                onClick={() => capture.capture.mutate()}
              >
                {capture.capture.isPending
                  ? "Requesting capture…"
                  : capture.id
                    ? `Capture Layer ${layer} Attention again`
                    : `Capture Layer ${layer} Attention`}
              </button>
            )}
          </div>
        </>
      ) : (
        <CapturedAttention
          key={capture.id}
          id={capture.id!}
          sourceId={sourceId}
          layer={layer}
          overview={capture.trace.data!.overview!}
          expires={capture.trace.data!.expires_at}
        />
      )}
    </div>
  );
}

function OutputAudit({ id, sourceId }: { id: string; sourceId: string }) {
  const comparison = useQuery({
    queryKey: ["output-comparison", id, sourceId],
    queryFn: ({ signal }) => api.compareOutputs(id, sourceId, signal),
    enabled: id !== sourceId,
  });
  if (id === sourceId)
    return (
      <p className="fine-print">
        Attention is retained in the current execution.
      </p>
    );
  return (
    <div
      className={`output-audit ${comparison.data && !comparison.data.passed ? "audit-warning" : ""}`}
    >
      {comparison.error ? (
        <>
          <p role="alert">
            Output comparison unavailable. The original execution may have
            expired; no match is assumed.
          </p>
          <ApiErrorNotice
            error={comparison.error}
            onRetry={() => void comparison.refetch()}
          />
        </>
      ) : comparison.isPending ? (
        <p role="status">Comparing retained final outputs…</p>
      ) : (
        <>
          <p role={comparison.data.passed ? undefined : "alert"}>
            {comparison.data.passed
              ? "Outputs match · same top token; full logits and probabilities within validated tolerance."
              : "Warning: these executions do not match within validated tolerance, or their execution settings differ."}
          </p>
          <RawData
            label="Output comparison and tolerances"
            data={comparison.data}
          />
        </>
      )}
    </div>
  );
}

function CapturedAttention({
  id,
  sourceId,
  layer,
  overview,
  expires,
}: {
  id: string;
  sourceId: string;
  layer: number;
  overview: TraceOverview;
  expires: string | null;
}) {
  const head = useSelection((s) => s.selectedHead) ?? 0;
  const query =
    useSelection((s) => s.selectedQueryToken) ??
    overview.input.sequence_length - 1;
  const key = useSelection((s) => s.selectedKeyToken) ?? query;
  const selectHead = useSelection((s) => s.selectHead);
  const selectRelation = useSelection((s) => s.selectRelation);
  const tokens = useQuery(tokensQuery(id));
  const layerData = useQuery(layerQuery(id, layer));
  const attention = useQuery({
    queryKey: [...traceKey(id), "attention", layer],
    queryFn: ({ signal }) => api.traceAttention(id, layer, signal),
  });
  const headData = useQuery({
    queryKey: [...traceKey(id), "head", layer, head],
    queryFn: ({ signal }) => api.traceHead(id, layer, head, {}, signal),
  });
  const architecture = overview.model.architecture;
  const queryToken = tokens.data?.[query];
  const keyToken = tokens.data?.[key];
  return (
    <>
      <LayerFlow
        index={layer}
        fallbackToken={query}
        data={layerData.data}
        insideAttention
      />
      <div className="execution-context">
        <span className="evidence-label">
          {id === sourceId
            ? "RETAINED EXECUTION"
            : "ATTENTION CAPTURE · SEPARATE EXECUTION"}
        </span>
        <details>
          <summary>Trace identities & capture details</summary>
          <dl className="facts">
            <div>
              <dt>Overview trace</dt>
              <dd>
                <code>{sourceId}</code>
              </dd>
            </div>
            <div>
              <dt>Attention trace</dt>
              <dd data-testid="detail-trace-id">
                <code>{id}</code>
              </dd>
            </div>
            <div>
              <dt>Capture / expires</dt>
              <dd>
                {overview.capture.profile} / {expires}
              </dd>
            </div>
            <div>
              <dt>Raw prompt</dt>
              <dd>{JSON.stringify(overview.input.text)}</dd>
            </div>
          </dl>
          <RawData
            label="Attention execution metadata"
            data={{
              model: overview.model,
              capture: overview.capture,
              validation: overview.validation,
            }}
          />
        </details>
        <OutputAudit id={id} sourceId={sourceId} />
      </div>
      <div className="attention-heading">
        <div>
          <p className="eyebrow">SELF ATTENTION / LAYER {layer}</p>
          <h4>Follow a token relationship.</h4>
        </div>
        <span className="evidence-label">
          RECONSTRUCTED · validated against captured eager weights
        </span>
      </div>
      <div
        className="head-controls"
        role="group"
        aria-label="Query head selection"
      >
        {Array.from({ length: architecture.query_heads }, (_, h) => (
          <button
            key={h}
            aria-label={`Query Head ${h}`}
            aria-pressed={head === h}
            onClick={() => selectHead(h)}
          >
            H{h}
          </button>
        ))}
      </div>
      <div className="head-context">
        <strong>
          Query Head {head} → KV Head {headData.data?.kv_head_index ?? "…"}
        </strong>
        <span>
          {architecture.head_dim} dimensions · {overview.input.sequence_length}{" "}
          tokens · {architecture.query_heads} query / {architecture.kv_heads} KV
          heads
        </span>
      </div>
      <p className="fine-print">
        Qwen3 uses grouped-query attention: multiple query heads share the same
        key/value head.
      </p>
      <div className="token-controls">
        <label>
          Query token
          <select
            aria-label="Query token"
            value={query}
            onChange={(e) => {
              const q = Number(e.target.value);
              selectRelation(q, q);
            }}
          >
            {tokens.data?.map((t) => (
              <option key={t.token_index} value={t.token_index}>
                {t.token_index} · {visibleWhitespace(t.decoded_token)}
              </option>
            ))}
          </select>
        </label>
        <span aria-hidden="true">→</span>
        <label>
          Key token
          <select
            aria-label="Key token"
            value={key}
            onChange={(e) => selectRelation(query, Number(e.target.value))}
          >
            {tokens.data?.map((t) => (
              <option key={t.token_index} value={t.token_index}>
                {t.token_index} · {visibleWhitespace(t.decoded_token)}
              </option>
            ))}
          </select>
        </label>
        <p className="fine-print">
          The query follows the selected input token. Choose a cell to inspect
          its key.
        </p>
      </div>
      {queryToken && keyToken && (
        <details className="token-identity">
          <summary>
            Exact token identities ·{" "}
            {visibleWhitespace(queryToken.decoded_token)} →{" "}
            {visibleWhitespace(keyToken.decoded_token)}
          </summary>
          <p>Query: {tokenDescription(queryToken)}</p>
          <p>Key: {tokenDescription(keyToken)}</p>
        </details>
      )}
      {tokens.error ? (
        <ApiErrorNotice
          error={tokens.error}
          onRetry={() => void tokens.refetch()}
        />
      ) : headData.error ? (
        <ApiErrorNotice
          error={headData.error}
          onRetry={() => void headData.refetch()}
        />
      ) : attention.error ? (
        <ApiErrorNotice
          error={attention.error}
          onRetry={() => void attention.refetch()}
        />
      ) : tokens.data && headData.data && attention.data ? (
        <>
          <div className="attention-reading">
            <AttentionHeatmap
              id={id}
              layer={layer}
              head={head}
              query={query}
              selectedKey={key}
              tokens={tokens.data}
            />
            <RelationInspector
              id={id}
              layer={layer}
              head={head}
              kvHead={headData.data.kv_head_index}
              query={query}
              selectedKey={key}
              tokens={tokens.data}
            />
          </div>
          <TokenConnections
            id={id}
            layer={layer}
            head={head}
            query={query}
            selectedKey={key}
            tokens={tokens.data}
          />
          <p className="attention-caveat">
            Attention weights describe internal routing, not causal attribution.
          </p>
          <details className="attention-glossary">
            <summary>Attention terms</summary>
            <dl>
              <dt>Query</dt>
              <dd>The selected position’s vector used to score keys.</dd>
              <dt>Key</dt>
              <dd>
                The vector against which a query is scored; shared across a
                group of query heads.
              </dd>
              <dt>Value</dt>
              <dd>
                The vector mixed into the head output according to attention
                weights.
              </dd>
              <dt>RoPE</dt>
              <dd>
                Rotary position encoding transforms Q and K before their dot
                product.
              </dd>
              <dt>Causal mask</dt>
              <dd>
                The native model mask prevents a query from attending to future
                positions.
              </dd>
              <dt>Attention weight</dt>
              <dd>
                A softmax weight normalized across the full allowed key row.
              </dd>
              <dt>GQA</dt>
              <dd>
                Grouped-query attention: multiple query heads share one
                key/value head.
              </dd>
            </dl>
          </details>
        </>
      ) : (
        <p role="status">Retrieving retained attention metadata…</p>
      )}
    </>
  );
}
