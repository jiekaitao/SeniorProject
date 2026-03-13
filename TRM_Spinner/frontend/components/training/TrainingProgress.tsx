"use client";

import { useEffect } from "react";
import GatorSpinner from "@/components/ui/GatorSpinner";
import LossCurve from "./LossCurve";
import MetricsPanel from "./MetricsPanel";
import { useTrainingProgress } from "@/hooks/useTrainingProgress";
import { fetchAPI } from "@/lib/api";

interface TrainingProgressProps {
  jobId: string;
  onComplete?: () => void;
}

export default function TrainingProgress({
  jobId,
  onComplete,
}: TrainingProgressProps) {
  const { metrics, status, latestMetrics } = useTrainingProgress(jobId);

  useEffect(() => {
    if (status === "completed" && onComplete) {
      onComplete();
    }
  }, [status, onComplete]);

  const handleCancel = async () => {
    try {
      await fetchAPI(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    } catch {
      // ignore cancel errors
    }
  };

  const statusText: Record<string, string> = {
    idle: "Waiting...",
    preparing: "Preparing model...",
    training: latestMetrics
      ? `Training step ${latestMetrics.step.toLocaleString()}`
      : "Starting training...",
    completed: "Training complete!",
    failed: "Training failed.",
    cancelled: "Training cancelled.",
  };

  return (
    <div className="border-t border-gator-200/50 bg-gator-50 p-6">
      <div className="mx-auto max-w-2xl space-y-6">
        <div className="flex flex-col items-center gap-3">
          {status !== "completed" &&
            status !== "failed" &&
            status !== "cancelled" && <GatorSpinner size={56} />}
          <p className="font-heading text-xl text-gator-500">
            {statusText[status] || "Processing..."}
          </p>
        </div>

        <MetricsPanel metrics={latestMetrics} />
        <LossCurve metrics={metrics} />

        {status === "training" && (
          <div className="flex justify-end">
            <button
              onClick={handleCancel}
              className="rounded-lg px-4 py-2 font-body text-sm text-gator-500/60 transition-colors hover:bg-gator-200/30 hover:text-gator-500"
            >
              Cancel training
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
