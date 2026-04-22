"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, Download, RefreshCw } from "lucide-react";

import ResearchBanner from "@/components/ui/ResearchBanner";
import GatorSpinner from "@/components/ui/GatorSpinner";
import { fetchAPI, fetchBlob } from "@/lib/api";
import GridView from "./GridView";

type SessionDetails = {
  id: string;
  state: string;
  job_id?: string | null;
  classification?: string | null;
  data_path?: string | null;
  created_at?: string;
  updated_at?: string;
};

type LatestMetrics = {
  job_id: string;
  status: string;
  metrics: Record<string, number | string>;
};

type PredictionRecord = {
  index: number;
  input: number[][];
  expected_output: number[][];
  predicted_output: number[][];
  exact_match: boolean;
  token_accuracy: number;
};

type PredictionsResponse = {
  summary: {
    total_examples: number;
    exact_match_rate: number;
    token_accuracy: number;
  };
  predictions: PredictionRecord[];
};

interface Props {
  sessionId: string;
}

const numberFmt = (v: unknown, digits = 4): string => {
  if (typeof v !== "number" || Number.isNaN(v)) return String(v ?? "—");
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  return v.toFixed(digits);
};

const pct = (v: number): string => {
  if (Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
};

export default function ResultsDashboard({ sessionId }: Props) {
  const router = useRouter();
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [latest, setLatest] = useState<LatestMetrics | null>(null);
  const [preds, setPreds] = useState<PredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const sess = await fetchAPI<SessionDetails>(`/api/sessions/${sessionId}`);
      setSession(sess);

      if (sess.job_id) {
        try {
          const metrics = await fetchAPI<LatestMetrics>(
            `/api/jobs/${sess.job_id}/latest`
          );
          setLatest(metrics);
        } catch {
          setLatest(null);
        }

        try {
          const p = await fetchAPI<PredictionsResponse>(
            `/api/jobs/${sess.job_id}/predictions`
          );
          setPreds(p);
        } catch {
          setPreds(null);
        }
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [sessionId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!session?.job_id) return;
    if (latest?.status === "completed") return;
    const id = setInterval(refresh, 2500);
    return () => clearInterval(id);
  }, [refresh, session?.job_id, latest?.status]);

  const downloadWeights = useCallback(async () => {
    if (!session?.job_id) return;
    try {
      const blob = await fetchBlob(`/api/jobs/${session.job_id}/download`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `trm_weights_${session.job_id.slice(0, 8)}.pt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // no-op
    }
  }, [session?.job_id]);

  if (loading) {
    return (
      <div className="flex min-h-screen flex-col">
        <ResearchBanner />
        <div className="flex flex-1 items-center justify-center">
          <GatorSpinner size={64} />
        </div>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="flex min-h-screen flex-col">
        <ResearchBanner />
        <div className="flex flex-1 items-center justify-center p-10">
          <div className="max-w-md rounded-xl bg-cream/80 p-8 text-center shadow-sm">
            <p className="font-heading text-2xl text-gator-600">
              Couldn&apos;t load this session
            </p>
            <p className="mt-2 font-body text-sm text-gator-500/70">
              {error || "The session was not found."}
            </p>
            <button
              onClick={() => router.push("/chat")}
              className="mt-6 rounded-lg bg-gator-500 px-5 py-2 font-body text-cream hover:bg-gator-600"
            >
              Back to sessions
            </button>
          </div>
        </div>
      </div>
    );
  }

  const m = latest?.metrics ?? {};
  const finalLoss = (m["train/lm_loss"] as number) ?? (m["loss"] as number);
  const finalAcc =
    (m["train/accuracy"] as number) ?? (m["accuracy"] as number);
  const finalExact = (m["train/exact_accuracy"] as number) ?? undefined;
  const step = m["step"];
  const totalSteps = m["total_steps"];

  return (
    <div className="flex min-h-screen flex-col bg-gator-50">
      <ResearchBanner />

      <div className="border-b border-gator-200/50 bg-cream/50 px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Link
              href={`/chat/${sessionId}`}
              className="rounded-lg p-1.5 text-gator-500/60 hover:bg-gator-100 hover:text-gator-500"
            >
              <ArrowLeft size={20} />
            </Link>
            <div>
              <h1 className="font-heading text-3xl text-gator-600">
                Training Results
              </h1>
              <p className="font-body text-xs text-gator-500/60">
                session{" "}
                <span className="font-mono text-gator-500/80">
                  {sessionId.slice(0, 8)}
                </span>{" "}
                · {session.classification || "unclassified"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={refresh}
              disabled={refreshing}
              className="flex items-center gap-1.5 rounded-lg border border-gator-200 bg-cream px-3 py-1.5 font-body text-sm text-gator-500 hover:bg-gator-100"
            >
              <RefreshCw
                size={14}
                className={refreshing ? "animate-spin" : ""}
              />
              Refresh
            </button>
            {session.job_id && latest?.status === "completed" && (
              <button
                onClick={downloadWeights}
                className="flex items-center gap-1.5 rounded-lg bg-gator-500 px-3 py-1.5 font-body text-sm text-cream hover:bg-gator-600"
              >
                <Download size={14} />
                Weights
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="mx-auto max-w-5xl space-y-8">
          {/* Metric summary */}
          <section className="rounded-2xl bg-cream/80 p-6 shadow-sm">
            <h2 className="font-heading text-2xl text-gator-600">
              Training summary
            </h2>
            <p className="mt-1 font-body text-sm text-gator-500/70">
              Status:{" "}
              <span className="font-mono text-gator-500">
                {latest?.status ?? "pending"}
              </span>
              {step !== undefined && totalSteps !== undefined && (
                <>
                  {" · step "}
                  <span className="font-mono">{String(step)}</span>/
                  <span className="font-mono">{String(totalSteps)}</span>
                </>
              )}
            </p>

            <div className="mt-6 grid gap-4 md:grid-cols-4">
              <MetricCard
                label="Final loss"
                value={numberFmt(finalLoss)}
                hint="train/lm_loss"
              />
              <MetricCard
                label="Token accuracy"
                value={typeof finalAcc === "number" ? pct(finalAcc) : "—"}
                hint="train/accuracy"
              />
              <MetricCard
                label="Exact match"
                value={typeof finalExact === "number" ? pct(finalExact) : "—"}
                hint="train/exact_accuracy"
              />
              <MetricCard
                label="Halt steps"
                value={numberFmt(m["train/steps"], 2)}
                hint="avg halt steps per batch"
              />
            </div>
          </section>

          {/* Predictions */}
          <section className="rounded-2xl bg-cream/80 p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="font-heading text-2xl text-gator-600">
                Sample predictions
              </h2>
              {preds && (
                <div className="font-body text-sm text-gator-500/70">
                  <span className="font-mono text-gator-500">
                    {pct(preds.summary.exact_match_rate)}
                  </span>{" "}
                  exact ·{" "}
                  <span className="font-mono text-gator-500">
                    {pct(preds.summary.token_accuracy)}
                  </span>{" "}
                  tokens
                </div>
              )}
            </div>

            {!preds && (
              <p className="mt-4 font-body text-sm text-gator-500/70">
                {latest?.status === "completed"
                  ? "Predictions are being generated, this page will refresh automatically."
                  : "Predictions appear here once training completes."}
              </p>
            )}

            {preds && preds.predictions.length === 0 && (
              <p className="mt-4 font-body text-sm text-gator-500/70">
                No predictions were written.
              </p>
            )}

            {preds && preds.predictions.length > 0 && (
              <div className="mt-6 space-y-4">
                {preds.predictions.slice(0, 8).map((rec) => (
                  <PredictionCard key={rec.index} rec={rec} />
                ))}
              </div>
            )}
          </section>

          {/* Raw metrics table */}
          <section className="rounded-2xl bg-cream/80 p-6 shadow-sm">
            <h2 className="font-heading text-2xl text-gator-600">
              Raw training metrics
            </h2>
            <p className="mt-1 font-body text-sm text-gator-500/70">
              Latest values published by the worker over Redis pub/sub.
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full font-mono text-sm">
                <tbody>
                  {Object.entries(m)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([k, v]) => (
                      <tr
                        key={k}
                        className="border-b border-gator-200/40 last:border-0"
                      >
                        <td className="py-1.5 pr-6 text-gator-500/70">{k}</td>
                        <td className="py-1.5 text-gator-600">
                          {typeof v === "number" ? numberFmt(v, 6) : String(v)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl bg-gator-50 p-4">
      <p className="font-body text-xs uppercase tracking-wide text-gator-500/60">
        {label}
      </p>
      <p className="mt-1 font-heading text-3xl text-gator-600">{value}</p>
      {hint && (
        <p className="mt-1 font-mono text-[11px] text-gator-500/50">{hint}</p>
      )}
    </div>
  );
}

function PredictionCard({ rec }: { rec: PredictionRecord }) {
  const badge = rec.exact_match ? "exact" : `${pct(rec.token_accuracy)} tokens`;
  const badgeClass = rec.exact_match
    ? "bg-gator-500 text-cream"
    : "bg-amber-accent/30 text-gator-600";

  return (
    <div className="rounded-xl bg-gator-50 p-4">
      <div className="flex items-center justify-between">
        <p className="font-body text-sm text-gator-500/80">
          Example #{rec.index}
        </p>
        <span
          className={`rounded-full px-2.5 py-0.5 font-mono text-[11px] ${badgeClass}`}
        >
          {badge}
        </span>
      </div>
      <div className="mt-3 grid gap-4 md:grid-cols-3">
        <LabeledGrid label="input" grid={rec.input} />
        <LabeledGrid label="expected" grid={rec.expected_output} />
        <LabeledGrid
          label="predicted"
          grid={rec.predicted_output}
          reference={rec.expected_output}
        />
      </div>
    </div>
  );
}

function LabeledGrid({
  label,
  grid,
  reference,
}: {
  label: string;
  grid: number[][];
  reference?: number[][];
}) {
  return (
    <div>
      <p className="mb-1 font-body text-[11px] uppercase tracking-wide text-gator-500/60">
        {label}
      </p>
      <GridView grid={grid} reference={reference} />
    </div>
  );
}
