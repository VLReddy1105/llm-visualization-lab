import { useQuery } from "@tanstack/react-query";
import { api, type TraceOverview } from "@/lib/api/client";
import { componentQuery, traceKey } from "@/lib/api/queries";
import { useSelection } from "@/state/selection";
import { Descriptor, RawData } from "@/components/ui/ScientificData";
import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";

export function ComponentInspector({
  id,
  component,
  overview,
}: {
  id: string;
  component: string;
  overview: TraceOverview;
}) {
  const selectedToken = useSelection((state) => state.selectedToken);
  const descriptor = useQuery(componentQuery(id, component));
  const slice = useQuery({
    queryKey: [...traceKey(id), "embedding-preview", selectedToken],
    queryFn: ({ signal }) =>
      api.traceSlice(
        id,
        component,
        {
          selectors: {
            batch: { index: 0 },
            token: { index: selectedToken! },
            hidden: {
              start: 0,
              stop: Math.min(64, overview.model.architecture.hidden_size),
              step: 1,
            },
          },
        },
        signal,
      ),
    enabled:
      component === "embedding.output" &&
      selectedToken != null &&
      descriptor.data?.raw_available === true,
  });
  if (descriptor.isPending)
    return <p role="status">Loading component descriptor…</p>;
  if (descriptor.error)
    return (
      <ApiErrorNotice
        error={descriptor.error}
        onRetry={() => void descriptor.refetch()}
      />
    );
  return (
    <div className="component-inspector">
      <div className="component-role">
        <p>
          {component === "embedding.output"
            ? "Maps each exact token ID to its learned representation."
            : component === "output.normalized"
              ? "Normalizes the final residual representation before vocabulary projection."
              : "Projects the final hidden representation into one score per vocabulary token."}
        </p>
        {component === "output.lm_head_input" && (
          <dl className="facts">
            <div>
              <dt>Vocabulary size</dt>
              <dd>
                {overview.model.architecture.vocabulary_size.toLocaleString(
                  "en",
                )}
              </dd>
            </div>
            <div>
              <dt>Output shape</dt>
              <dd>1 × 1 × {overview.model.architecture.vocabulary_size}</dd>
            </div>
            <div>
              <dt>Prediction position</dt>
              <dd>{overview.prediction.prediction_position}</dd>
            </div>
          </dl>
        )}
        {component === "embedding.output" && selectedToken != null && (
          <div className="vector-preview">
            <strong>Selected token {selectedToken}</strong>
            {!descriptor.data.raw_available ? (
              <p>Not captured in this trace</p>
            ) : slice.error ? (
              <ApiErrorNotice error={slice.error} />
            ) : slice.isPending ? (
              <p role="status">Retrieving bounded embedding preview…</p>
            ) : (
              <>
                <p>
                  {slice.data.shape.join(" × ")} · hidden dimensions 0–
                  {slice.data.values.length - 1}
                </p>
                <p className="fine-print">
                  A bounded preview of the captured vector. No semantic meaning
                  is assigned to individual coordinates.
                </p>
                <RawData label="Embedding preview values" data={slice.data} />
              </>
            )}
          </div>
        )}
      </div>
      <Descriptor data={descriptor.data} />
    </div>
  );
}
