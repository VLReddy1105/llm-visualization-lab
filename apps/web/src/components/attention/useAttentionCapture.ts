import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, type TraceOverview } from "@/lib/api/client";
import { overviewQuery, traceKey } from "@/lib/api/queries";
import { useSelection, useSelectionStore } from "@/state/selection";

export function useAttentionCapture(
  sourceId: string,
  layer: number,
  source: TraceOverview,
  profile: "attention" | "overview" | "mlp" = "attention",
  token?: number,
) {
  const [enteredAt] = useState(Date.now);
  const store = useSelectionStore();
  const cache = useQueryClient();
  const key = `${sourceId}:${layer}:${profile}${profile === "mlp" ? `:${token}` : ""}`;
  const associated = useSelection((s) =>
    profile === "attention" ? s.attentionTraces[key] : profile === "mlp" ? s.blockTraces[key] : s.evolutionTraces[key],
  );
  const available =
    profile === "attention"
      ? source.available.attention_layers.includes(layer)
      : profile === "mlp"
        ? source.available.mlp_layers.includes(layer) && token != null && source.capture.token_indices.includes(token)
        : source.capture.profile === "overview" &&
        source.capture.token_indices.length === source.input.sequence_length;
  // An explicit recapture can upgrade an older MLP source whose vectors exist
  // but whose original learned-module validation references are missing.
  const id = profile === "mlp"
    ? associated ?? (available ? sourceId : undefined)
    : available ? sourceId : associated;
  const issue = useSelection((s) => (id ? s.detailIssues[id] : undefined));
  const [timedOutId, timeOut] = useState<string | null>(null);
  const options = overviewQuery(id ?? null);
  const trace = useQuery({
    ...options,
    enabled: !!id && !issue,
    refetchInterval:
      timedOutId === id || issue ? false : options.refetchInterval,
  });
  const expires = trace.data?.expires_at;
  const pending = ["queued", "running"].includes(trace.data?.status ?? "");
  useEffect(() => {
    if (!id || !expires || issue) return;
    const expire = () => {
      if (Date.now() >= Date.parse(expires)) {
        store
          .getState()
          .setDetailIssue(
            id,
            new ApiError(
              "The retained detailed execution has expired. Capture again to inspect it.",
              "TRACE_EXPIRED",
              410,
            ),
          );
        void cache.cancelQueries({ queryKey: traceKey(id) });
        cache.removeQueries({ queryKey: traceKey(id) });
      }
    };
    const timer = setTimeout(
      expire,
      Math.max(0, Date.parse(expires) - Date.now()),
    );
    window.addEventListener("focus", expire);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("focus", expire);
    };
  }, [id, expires, issue, store, cache]);
  useEffect(() => {
    if (!id || !pending || !trace.data?.created_at) return;
    const timer = setTimeout(
      () => timeOut(id),
      Math.max(0, Date.parse(trace.data.created_at) + 120000 - Date.now()),
    );
    return () => clearTimeout(timer);
  }, [id, pending, trace.data?.created_at]);
  const capture = useMutation({
    mutationFn: async () => {
      const ownerId = store.getState().traceId;
      const data = await api.createTrace({
        prompt: source.input.text,
        capture:
          profile === "mlp"
            ? { profile, layers: [layer], tokens: token == null ? undefined : [token] }
            : profile === "attention" ? { profile, layers: [layer] } : { profile },
        display_top_k: 10,
      });
      return { data, ownerId };
    },
    onSuccess: ({ data, ownerId }) => {
      if (store.getState().traceId !== ownerId) return;
      cache.setQueryData(overviewQuery(data.trace_id).queryKey, data);
      if (profile === "attention")
        store.getState().associateAttention(key, data.trace_id);
      else if (profile === "mlp")
        store.getState().associateBlock(key, data.trace_id);
      else store.getState().associateEvolution(key, data.trace_id);
    },
  });
  const cancel = useMutation({
    mutationFn: () => api.deleteTrace(id!),
    onSuccess: () => {
      if (id)
        store
          .getState()
          .setDetailIssue(
            id,
            new ApiError(
              "Detailed capture was cancelled. You can capture again explicitly.",
              "TRACE_CANCELLED",
            ),
          );
    },
  });
  const expired = !!expires && Date.parse(expires) <= enteredAt;
  return {
    id,
    trace,
    issue,
    capture,
    cancel,
    pending,
    timedOut: timedOutId === id,
    ready:
      trace.data?.status === "ready" &&
      !!trace.data.overview &&
      !issue &&
      !expired,
  };
}
