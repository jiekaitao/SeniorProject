"use client";

import AdminGuard from "@/components/auth/AdminGuard";
import ResearchBanner from "@/components/ui/ResearchBanner";

export default function DevLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AdminGuard>
      <div className="flex min-h-screen flex-col">
        <ResearchBanner />
        <div className="flex-1">{children}</div>
      </div>
    </AdminGuard>
  );
}
