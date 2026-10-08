import { ApiError } from "@/lib/api/client";
import { RawData } from "@/components/ui/ScientificData";

const titles: Record<string, string> = {
  TRACE_EXPIRED: "Trace expired",
  TRACE_EVICTED: "Trace evicted",
  TRACE_NOT_FOUND: "Trace not found",
  TRACE_DELETED: "Trace deleted",
  COMPONENT_NOT_CAPTURED: "Not captured in this trace",
  MODEL_NOT_READY: "Model is not ready",
  CAPTURE_BUDGET_EXCEEDED: "Capture budget exceeded",
  TRACE_STORE_BUDGET_EXCEEDED: "Trace store budget exceeded",
  API_UNAVAILABLE: "API unavailable",
  INVALID_REQUEST: "Check your prompt",
  TRACE_TIMEOUT: "Trace is taking longer than expected",
};
export function ApiErrorNotice({
  error,
  onRetry,
  retryLabel = "Try again",
}: {
  error: Error;
  onRetry?: () => void;
  retryLabel?: string;
}) {
  const apiError = error instanceof ApiError ? error : null;
  return (
    <div
      className={`error-notice ${apiError?.code === "COMPONENT_NOT_CAPTURED" ? "missing-notice" : ""}`}
      role="alert"
    >
      <div>
        <strong>
          {apiError
            ? (titles[apiError.code] ?? "Trace request failed")
            : "Request failed"}
        </strong>
        <p>{error.message}</p>
        {apiError && (
          <span className="fine-print">
            {apiError.code}
            {apiError.requestId ? ` · Request ${apiError.requestId}` : ""}
          </span>
        )}
      </div>
      {onRetry && (
        <button className="secondary-button" onClick={onRetry}>
          {retryLabel}
        </button>
      )}
      {apiError?.details && (
        <RawData label="Error details" data={apiError.details} />
      )}
    </div>
  );
}
