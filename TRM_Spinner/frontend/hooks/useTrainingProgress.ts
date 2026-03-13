"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { createSSEConnection } from "@/lib/api";

export interface TrainingMetric {
  step: number;
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

        if (data.type === "metric") {
          const metric = data.data as TrainingMetric;
          setMetrics((prev) => [...prev, metric]);
          setStatus("training");
        } else if (data.type === "status") {
          setStatus(data.status as TrainingStatus);
        } else if (data.type === "complete") {
          setStatus("completed");
          disconnect();
        } else if (data.type === "error") {
          setStatus("failed");
          disconnect();
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
