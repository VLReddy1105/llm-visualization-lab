import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { SliceRequest } from "@/lib/api/client";
import { RawData } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { attentionComponent, exact, sliceQuery } from "./data";

function Vector({
  id,
  component,
  selectors,
  name,
}: {
  id: string;
  component: string;
  selectors: SliceRequest["selectors"];
  name: string;
}) {
  const vector = useQuery(sliceQuery(id, component, selectors, true));
  if (vector.error)
    return (
      <ApiErrorNotice
        error={vector.error}
        onRetry={() => void vector.refetch()}
      />
    );
  if (!vector.data) return <p role="status">Loading {name} vector…</p>;
  const data = vector.data,
    stats = data.statistics;
  // Pixel scaling only. Every scientific statistic comes from the bounded API slice.
  const magnitude =
    Math.max(Math.abs(stats?.min ?? 0), Math.abs(stats?.max ?? 0)) || 1;
  return (
    <section className="vector-inspector" aria-label={`${name} vector`}>
      <div className="section-heading">
        <strong>{name}</strong>
        <span className="evidence-label">{data.provenance.toUpperCase()}</span>
      </div>
      <p className="fine-print">
        [{data.values.length}] · signed dimensions · ±{magnitude.toPrecision(4)}
      </p>
      <svg
        viewBox={`0 0 ${data.values.length * 3} 90`}
        role="img"
        aria-label={`${name}: ${data.values.length} signed coordinates; positive above zero, negative below. Raw values available below.`}
      >
        <line
          x1="0"
          y1="45"
          x2={data.values.length * 3}
          y2="45"
          stroke="#8b98a9"
        />
        {data.values.map((v, i) =>
          v == null ? null : (
            <line
              key={i}
              x1={i * 3 + 1}
              x2={i * 3 + 1}
              y1="45"
              y2={45 - (v / magnitude) * 40}
              stroke={v < 0 ? "#8b658b" : "#386d9c"}
              strokeWidth="2"
            >
              <title>
                Dimension {i}: {exact(v)}
              </title>
            </line>
          ),
        )}
      </svg>
      {stats ? (
        <>
          <dl className="vector-statistics">
            {(["l2_norm", "min", "max", "mean", "std"] as const).map(
              (field) => (
                <div key={field}>
                  <dt>{field === "l2_norm" ? "L2 norm" : field}</dt>
                  <dd
                    data-testid={`${name.split(" ")[0].toLowerCase()}-${field}`}
                  >
                    {exact(stats[field])}
                  </dd>
                </div>
              ),
            )}
          </dl>
          <p className="fine-print">
            SUMMARY · {stats.accumulation_dtype} · population std · selected
            vector only.
          </p>
        </>
      ) : (
        <p>Selected-vector statistics were not provided.</p>
      )}
      <RawData label={`Show raw values · ${name}`} data={data} />
    </section>
  );
}
export function VectorInspector({
  id,
  layer,
  head,
  kvHead,
  query,
  selectedKey,
}: {
  id: string;
  layer: number;
  head: number;
  kvHead: number;
  query: number;
  selectedKey: number;
}) {
  const [stage, setStage] = useState("rotary");
  const q = {
    batch: { index: 0 },
    query_head: { index: head },
    query_token: { index: query },
  };
  const kv = {
    batch: { index: 0 },
    kv_head: { index: kvHead },
    key_token: { index: selectedKey },
  };
  const label =
    stage === "rotary"
      ? "after RoPE"
      : stage === "normalized"
        ? "after Q/K norm"
        : "projected";
  return (
    <div className="qkv-panel">
      <label className="vector-stage">
        Q / K observation
        <select
          aria-label="Q / K observation stage"
          value={stage}
          onChange={(e) => setStage(e.target.value)}
        >
          <option value="rotary">After norm + RoPE · score operands</option>
          <option value="normalized">After Q/K normalization</option>
          <option value="projected">
            Projected · before Q/K normalization
          </option>
        </select>
      </label>
      <p className="fine-print">
        Q and K use independent signed plot scales. Only post-norm, post-RoPE
        vectors produce the retained Q · K score above.
      </p>
      <div className="qk-vectors">
        <Vector
          id={id}
          component={attentionComponent(layer, `query.${stage}`)}
          selectors={q}
          name={`Q ${label}`}
        />
        <Vector
          id={id}
          component={attentionComponent(layer, `key.${stage}`)}
          selectors={kv}
          name={`K ${label}`}
        />
      </div>
      <Vector
        id={id}
        component={attentionComponent(layer, "value")}
        selectors={kv}
        name="V projected"
      />
      <p className="fine-print">
        The attention weight controls how strongly this value vector contributes
        to the head output. V bypasses Q/K normalization and RoPE.
      </p>
      <details className="head-output">
        <summary>Weighted value mixture / head output</summary>
        <HeadOutput
          id={id}
          component={attentionComponent(layer, "head_output")}
          selectors={q}
        />
      </details>
    </div>
  );
}
function HeadOutput(props: {
  id: string;
  component: string;
  selectors: SliceRequest["selectors"];
}) {
  return <Vector {...props} name="Head output" />;
}
