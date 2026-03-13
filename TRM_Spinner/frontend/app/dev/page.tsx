"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Activity, Users, Cpu, Database } from "lucide-react";
import GatorSpinner from "@/components/ui/GatorSpinner";
import { fetchAPI } from "@/lib/api";

interface Stats {
  total_users: number;
  total_jobs: number;
  active_jobs: number;
  completed_jobs: number;
}

function StatCard({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  icon: React.ComponentType<{ size?: number; className?: string }>;
}) {
  return (
    <div className="rounded-2xl bg-cream p-6 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-body text-sm text-gator-600/60">{label}</p>
          <p className="mt-2 font-heading text-3xl text-gator-500">{value}</p>
        </div>
        <Icon size={24} className="text-gator-500/30" />
      </div>
    </div>
  );
}

export default function DevDashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAPI<Stats>("/api/admin/stats")
      .then(setStats)
      .catch(() =>
        setStats({
          total_users: 0,
          total_jobs: 0,
          active_jobs: 0,
          completed_jobs: 0,
        })
      )
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <GatorSpinner size={64} />
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="mx-auto max-w-4xl">
        <div className="mb-8 flex items-center justify-between">
          <h1 className="font-heading text-3xl text-gator-600">
            Dev Dashboard
          </h1>
          <Link
            href="/dev/jobs"
            className="rounded-xl bg-gator-500 px-4 py-2 font-body text-sm text-cream transition-colors hover:bg-gator-600"
          >
            View Jobs
          </Link>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Users"
            value={stats?.total_users ?? 0}
            icon={Users}
          />
          <StatCard
            label="Total Jobs"
            value={stats?.total_jobs ?? 0}
            icon={Database}
          />
          <StatCard
            label="Active Jobs"
            value={stats?.active_jobs ?? 0}
            icon={Cpu}
          />
          <StatCard
            label="Completed"
            value={stats?.completed_jobs ?? 0}
            icon={Activity}
          />
        </div>
      </div>
    </div>
  );
}
