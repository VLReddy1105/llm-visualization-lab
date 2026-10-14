import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type TokenMetadata } from "@/lib/api/client";
import { traceKey } from "@/lib/api/queries";
import { RawData, visibleWhitespace } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { exact } from "./data";
import { VectorInspector } from "./VectorInspector";

export function RelationInspector({
  id,
  layer,
  head,
  kvHead,
  query,
  selectedKey,
  tokens,
}: {
  id: string;
  layer: number;
  head: number;
  kvHead: number;
  query: number;
  selectedKey: number;
  tokens: TokenMetadata[];
}) {
  const [vectors, showVectors] = useState(false);
  const relation = useQuery({
    queryKey: [...traceKey(id), "relation", layer, head, query, selectedKey],
    queryFn: ({ signal }) =>
      api.traceRelation(id, layer, head, query, selectedKey, signal),
  });
  return (
    <section className="relation-inspector" aria-labelledby="relation-heading">
      <p className="eyebrow">MATHEMATICAL RELATION · LAYER {layer}</p>
      <h5 id="relation-heading">
        {visibleWhitespace(tokens[query].decoded_token)} <span>→</span>{" "}
        {visibleWhitespace(tokens[selectedKey].decoded_token)}
      </h5>
      <p className="fine-print">
        Query {query} / head {head} · key {selectedKey} / KV head {kvHead}
      </p>
      {relation.error ? (
        <ApiErrorNotice
          error={relation.error}
          onRetry={() => void relation.refetch()}
        />
      ) : !relation.data ? (
        <p role="status">Retrieving retained relation…</p>
      ) : relation.data.masked ? (
        <p className="masked-relation" role="status">
          Causally masked · this key is unavailable to the selected query. It is
          not a valid attention relation.
        </p>
      ) : (
        <>
          <p className="relation-operands">
            Q projection → Q norm → RoPE
            <br />K projection → K norm → RoPE
          </p>
          <ol className="math-steps">
            <li>
              <div>
                <strong>Q · K</strong>
                <span>DERIVED · post-norm, post-RoPE operands</span>
              </div>
              <output data-testid="qk-score">
                {exact(relation.data.observations.similarity.values[0])}
              </output>
            </li>
            <li>
              <div>
                <strong>Scale by 1 / √head dimension</strong>
                <span>Adapter scaling factor</span>
              </div>
              <output data-testid="scaling">
                {exact(relation.data.scaling)}
              </output>
            </li>
            <li>
              <div>
                <strong>Scaled score</strong>
                <span>DERIVED</span>
              </div>
              <output data-testid="scaled-score">
                {exact(relation.data.observations.scaled_scores.values[0])}
              </output>
            </li>
            <li>
              <div>
                <strong>Causal mask</strong>
                <span>Retained native mask</span>
              </div>
              <output>Allowed</output>
            </li>
            <li className="math-result">
              <div>
                <strong>Attention weight</strong>
                <span>RECONSTRUCTED · full-row softmax</span>
              </div>
              <output data-testid="relation-weight">
                {exact(
                  relation.data.observations.weights_reconstructed.values[0],
                )}
              </output>
            </li>
          </ol>
          <p className="fine-print">
            Softmax uses the full allowed key row, including keys outside the
            visible matrix window. These scalars are retained backend
            observations.
          </p>
          <RawData
            label="Relation provenance & exact values"
            data={relation.data}
          />
          <button
            className="secondary-button vector-toggle"
            aria-expanded={vectors}
            onClick={() => showVectors(!vectors)}
          >
            {vectors ? "Hide Q / K / V vectors" : "Inspect Q / K / V vectors"}
          </button>
          {vectors && (
            <VectorInspector
              id={id}
              layer={layer}
              head={head}
              kvHead={kvHead}
              query={query}
              selectedKey={selectedKey}
            />
          )}
        </>
      )}
    </section>
  );
}
