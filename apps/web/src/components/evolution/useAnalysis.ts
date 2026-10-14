import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, type AnalysisRequest } from "@/lib/api/client";

/** Identical POSTs are backend-deduplicated saved-data jobs, never trace executions. */
export function useAnalysis(id: string, request: AnalysisRequest) {
  const cache = useQueryClient();
  const [enteredAt] = useState(Date.now);
  const key = ["trace", id, "analysis-request", request];
  const start = useQuery({
    queryKey: key,
    queryFn: ({ signal }) => api.createAnalysis(id, request, signal),
  });
  const ack = start.data;
  const [expiredId, expire] = useState<string | null>(null);
  const [timedOutId, timeOut] = useState<string | null>(null);
  const jobKey = ["trace", id, "analysis-job", ack?.analysis_id];
  const job = useQuery({
    queryKey: jobKey,
    queryFn: ({ signal }) => api.analysis(id, ack!.analysis_id, signal),
    enabled: !!ack && expiredId !== ack.analysis_id,
    initialData: ack,
    refetchInterval: (q) =>
      q.state.data &&
      ["queued", "running"].includes(q.state.data.status) &&
      timedOutId !== q.state.data.analysis_id &&
      expiredId !== q.state.data.analysis_id
        ? 300
        : false,
  });
  const data = job.data;
  useEffect(() => {
    if (!data) return;
    const check = () => {
      if (Date.now() >= Date.parse(data.expires_at)) expire(data.analysis_id);
    };
    const timer = setTimeout(
      check,
      Math.max(0, Date.parse(data.expires_at) - Date.now()),
    );
    window.addEventListener("focus", check);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("focus", check);
    };
  }, [data]);
  useEffect(() => {
    if (!data || !["queued", "running"].includes(data.status)) return;
    const timer = setTimeout(
      () => timeOut(data.analysis_id),
      Math.max(0, Date.parse(data.created_at) + 120000 - Date.now()),
    );
    return () => clearTimeout(timer);
  }, [data]);
  const cancel = useMutation({
    mutationFn: () => api.cancelAnalysis(id, ack!.analysis_id),
    onSuccess: (response) => cache.setQueryData(jobKey, response),
  });
  const expired =
    !!data &&
    (expiredId === data.analysis_id ||
      Date.parse(data.expires_at) <= enteredAt);
  const error =
    start.error ??
    job.error ??
    cancel.error ??
    (expired
      ? new ApiError(
          "This analysis expired. Recompute from retained data; no model forward is required.",
          "ANALYSIS_EXPIRED",
        )
      : data?.error
        ? new ApiError(
            data.error.message,
            data.error.code,
            undefined,
            data.error.details,
          )
        : data?.status === "cancelled"
          ? new ApiError("Analysis was cancelled.", "ANALYSIS_CANCELLED")
          : null);
  return {
    data,
    error,
    ready: !error && data?.status === "ready",
    cancel,
    timedOut: data?.analysis_id === timedOutId,
    check: () => job.refetch(),
    retry: () => {
      cancel.reset();
      cache.removeQueries({ queryKey: jobKey, exact: true });
      return start.refetch();
    },
  };
}
