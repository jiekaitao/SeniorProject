"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { createSSEConnection } from "@/lib/api";

export interface TrainingMetric {
  step: number;
  total_steps?: number;
  loss: number;
  accuracy?: number;
  learning_rate?: number;
  elapsed_seconds?: number;
}

export type TrainingStatus =
  | "idle"
  | "preparing"
  | "training"
  | "completed"
  | "failed"
  | "cancelled";

/**
 * Normalise a raw Redis metrics payload into the shape our components expect.
 *
 * The worker publishes two kinds of messages on `trm:jobs:{id}:metrics`:
 *  - `{status: "started"|"completed"|"failed"|"cancelled", ...}` — lifecycle.
 *  - `{step, total_steps, "train/lm_loss", "train/accuracy", "train/lr"}` — metrics.
 *
 * We also accept the already-wrapped `{type, data}` form in case the backend
 * pre-normalises (e.g. in tests or future code paths).
 */
function normalise(raw: Record<string, unknown>): {
  kind: "metric" | "status" | "none";
  metric?: TrainingMetric;
  status?: TrainingStatus;
} {
  // Pre-wrapped forms (legacy/test path).
  if (typeof raw.type === "string") {
    const t = raw.type;
    if (t === "metric" && raw.data && typeof raw.data === "object") {
      return { kind: "metric", metric: raw.data as TrainingMetric };
    }
    if (t === "status" && typeof raw.status === "string") {
      return { kind: "status", status: raw.status as TrainingStatus };
    }
    if (t === "complete") return { kind: "status", status: "completed" };
    if (t === "error") return { kind: "status", status: "failed" };
  }

  // Lifecycle pings.
  if (typeof raw.status === "string") {
    const s = raw.status;
    if (s === "started") return { kind: "status", status: "training" };
    if (s === "completed") return { kind: "status", status: "completed" };
    if (s === "failed") return { kind: "status", status: "failed" };
    if (s === "cancelled") return { kind: "status", status: "cancelled" };
    if (s === "training") return { kind: "status", status: "training" };
  }

  // Raw metric dicts from pretrain_web.
  if (typeof raw.step === "number") {
    const loss =
      (raw["train/lm_loss"] as number | undefined) ??
      (raw["loss"] as number | undefined) ??
      0;
    const accuracy =
      (raw["train/accuracy"] as number | undefined) ??
      (raw["train/exact_accuracy"] as number | undefined) ??
      (raw["accuracy"] as number | undefined);
    const learningRate =
      (raw["train/lr"] as number | undefined) ??
      (raw["learning_rate"] as number | undefined);
    const totalSteps = raw["total_steps"] as number | undefined;
    return {
      kind: "metric",
      metric: {
        step: raw.step,
        total_steps: totalSteps,
        loss,
        accuracy,
        learning_rate: learningRate,
      },
    };
  }

  return { kind: "none" };
}

export function useTrainingProgress(jobId: string | null) {
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [status, setStatus] = useState<TrainingStatus>("idle");
  const [isConnected, setIsConnected] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  const disconnect = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;

    setMetrics([]);
    setStatus("preparing");

    const source = createSSEConnection(
      `/api/jobs/${jobId}/stream`,
      (data) => {
        setIsConnected(true);

        const parsed = normalise(data);
        if (parsed.kind === "metric" && parsed.metric) {
          setMetrics((prev) => [...prev, parsed.metric!]);
          setStatus((s) => (s === "completed" || s === "failed" ? s : "training"));
        } else if (parsed.kind === "status" && parsed.status) {
          setStatus(parsed.status);
          if (
            parsed.status === "completed" ||
            parsed.status === "failed" ||
            parsed.status === "cancelled"
          ) {
            disconnect();
          }
        }
      },
      () => {
        setIsConnected(false);
      }
    );

    sourceRef.current = source;

    return () => {
      disconnect();
    };
  }, [jobId, disconnect]);

  const latestMetrics = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  return { metrics, status, latestMetrics, isConnected, disconnect };
}
