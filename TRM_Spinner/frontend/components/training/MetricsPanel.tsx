"use client";

import type { TrainingMetric } from "@/hooks/useTrainingProgress";

interface MetricsPanelProps {
  metrics: TrainingMetric | null;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-xl bg-cream p-4 shadow-sm">
      <p className="font-body text-xs text-gator-600/60">{label}</p>
      <p className="mt-1 font-heading text-2xl text-gator-500">{value}</p>
    </div>
  );
}

export default function MetricsPanel({ metrics }: MetricsPanelProps) {
  if (!metrics) return null;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      <MetricCard label="Step" value={metrics.step.toLocaleString()} />
      <MetricCard label="Loss" value={metrics.loss.toFixed(4)} />
      {metrics.accuracy != null && (
        <MetricCard
          label="Accuracy"
          value={`${(metrics.accuracy * 100).toFixed(1)}%`}
        />
      )}
      {metrics.learning_rate != null && (
        <MetricCard
          label="Learning Rate"
          value={metrics.learning_rate.toExponential(2)}
        />
      )}
      {metrics.elapsed_seconds != null && (
        <MetricCard
          label="Elapsed"
          value={formatTime(metrics.elapsed_seconds)}
        />
      )}
    </div>
  );
}
