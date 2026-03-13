"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { TrainingMetric } from "@/hooks/useTrainingProgress";

interface LossCurveProps {
  metrics: TrainingMetric[];
}

export default function LossCurve({ metrics }: LossCurveProps) {
  if (metrics.length < 2) return null;

  return (
    <div className="rounded-2xl bg-cream p-4 shadow-sm">
      <h3 className="mb-3 font-heading text-lg text-gator-600">
        Training Loss
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={metrics}>
          <CartesianGrid stroke="#2d5a3d15" strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 12, fontFamily: "var(--font-patrick-hand)" }}
            stroke="#2d5a3d60"
          />
          <YAxis
            tick={{ fontSize: 12, fontFamily: "var(--font-patrick-hand)" }}
            stroke="#2d5a3d60"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#f5f0e8",
              border: "1px solid #c8e6c9",
              borderRadius: "12px",
              fontFamily: "var(--font-patrick-hand)",
            }}
          />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#2d5a3d"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#2d5a3d" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
