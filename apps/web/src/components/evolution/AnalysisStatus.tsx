import { ApiErrorNotice } from "@/components/trace/ApiErrorNotice";
import { useAnalysis } from "./useAnalysis";

export function AnalysisStatus({
  analysis,
}: {
  analysis: ReturnType<typeof useAnalysis>;
}) {
  if (analysis.error)
    return (
      <ApiErrorNotice
        error={analysis.error}
        onRetry={() => void analysis.retry()}
        retryLabel="Recompute saved-data analysis"
      />
    );
  if (analysis.ready) return null;
  return (
    <div className="analysis-status" role="status">
      <p>
        {analysis.timedOut
          ? "Analysis has not completed within two minutes."
          : "Analyzing saved observations…"}
        {analysis.data &&
          ` ${analysis.data.completed_checkpoints} / ${analysis.data.total_checkpoints} checkpoints · ${analysis.data.status}`}
      </p>
      {analysis.timedOut && (
        <button
          className="secondary-button"
          onClick={() => void analysis.check()}
        >
          Check analysis status
        </button>
      )}
      {analysis.data && (
        <button
          className="text-button"
          onClick={() => analysis.cancel.mutate()}
          disabled={analysis.cancel.isPending}
        >
          Cancel analysis
        </button>
      )}
    </div>
  );
}
