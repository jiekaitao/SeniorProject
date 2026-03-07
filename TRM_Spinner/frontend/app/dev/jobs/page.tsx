"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import GatorSpinner from "@/components/ui/GatorSpinner";
import { fetchAPI } from "@/lib/api";

interface Job {
  id: string;
  user_id: string;
  status: string;
  created_at: string;
  config: {
    model_type?: string;
    total_steps?: number;
  };
  current_step?: number;
  final_loss?: number;
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    queued: "bg-gator-50 text-gator-500/60",
    preparing: "bg-amber-accent/20 text-amber-accent",
    training: "bg-gator-100 text-gator-500",
    completed: "bg-gator-200 text-gator-600",
    failed: "bg-red-50 text-red-600",
    cancelled: "bg-gator-50 text-gator-500/40",
  };

  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 font-body text-xs ${
        styles[status] || styles.queued
      }`}
    >
      {status}
    </span>
  );
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAPI<{ jobs: Job[] }>("/api/admin/jobs")
      .then((res) => setJobs(res.jobs))
      .catch(() => setJobs([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-8">
      <div className="mx-auto max-w-5xl">
        <div className="mb-8 flex items-center gap-4">
          <Link
            href="/dev"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-gator-500/60 transition-colors hover:bg-gator-200/30 hover:text-gator-500"
          >
            <ArrowLeft size={18} />
          </Link>
          <h1 className="font-heading text-3xl text-gator-600">All Jobs</h1>
        </div>

        {loading ? (
          <div className="flex justify-center py-12">
            <GatorSpinner size={48} />
          </div>
        ) : jobs.length === 0 ? (
          <p className="py-12 text-center font-body text-gator-500/50">
            No jobs found
          </p>
        ) : (
          <div className="overflow-hidden rounded-2xl bg-cream shadow-sm">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gator-200/50">
                  <th className="px-5 py-3 text-left font-body text-xs font-normal text-gator-600/60">
                    Job ID
                  </th>
                  <th className="px-5 py-3 text-left font-body text-xs font-normal text-gator-600/60">
                    Status
                  </th>
                  <th className="px-5 py-3 text-left font-body text-xs font-normal text-gator-600/60">
                    Model
                  </th>
                  <th className="px-5 py-3 text-left font-body text-xs font-normal text-gator-600/60">
                    Progress
                  </th>
                  <th className="px-5 py-3 text-left font-body text-xs font-normal text-gator-600/60">
                    Loss
                  </th>
                  <th className="px-5 py-3 text-left font-body text-xs font-normal text-gator-600/60">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr
                    key={job.id}
                    className="border-b border-gator-200/30 last:border-0"
                  >
                    <td className="px-5 py-3 font-mono text-xs text-gator-500">
                      {job.id.slice(0, 8)}
                    </td>
                    <td className="px-5 py-3">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-5 py-3 font-body text-sm text-gator-600">
                      {job.config.model_type || "trm"}
                    </td>
                    <td className="px-5 py-3 font-body text-sm text-gator-500/70">
                      {job.current_step != null && job.config.total_steps
                        ? `${job.current_step.toLocaleString()} / ${job.config.total_steps.toLocaleString()}`
                        : "--"}
                    </td>
                    <td className="px-5 py-3 font-body text-sm text-gator-500/70">
                      {job.final_loss != null
                        ? job.final_loss.toFixed(4)
                        : "--"}
                    </td>
                    <td className="px-5 py-3 font-body text-xs text-gator-500/50">
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
