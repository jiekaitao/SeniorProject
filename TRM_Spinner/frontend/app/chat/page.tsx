"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Plus, MessageSquare, LogOut } from "lucide-react";
import AuthGuard from "@/components/auth/AuthGuard";
import ResearchBanner from "@/components/ui/ResearchBanner";
import GatorSpinner from "@/components/ui/GatorSpinner";
import { fetchAPI } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";

interface Session {
  id: string;
  title: string;
  created_at: string;
  status: string;
}

function ChatListContent() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const { user, logout } = useAuth();

  useEffect(() => {
    fetchAPI<{
      documents: Array<{
        $id: string;
        title: string;
        created_at: string;
        status: string;
      }>;
      total: number;
    }>("/api/sessions")
      .then((res) =>
        setSessions(
          (res.documents || []).map((d) => ({
            id: d.$id,
            title: d.title,
            created_at: d.created_at,
            status: d.status,
          }))
        )
      )
      .catch(() => setSessions([]))
      .finally(() => setLoading(false));
  }, []);

  const handleNewChat = async () => {
    try {
      const res = await fetchAPI<{ id: string }>("/api/sessions", {
        method: "POST",
        body: JSON.stringify({ user_id: user?.$id }),
      });
      router.push(`/chat/${res.id}`);
    } catch {
      const id = crypto.randomUUID();
      router.push(`/chat/${id}`);
    }
  };

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <div className="flex h-screen flex-col">
      <ResearchBanner />

      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="border-b border-gator-200/50 bg-cream/30 p-6">
          <div className="mx-auto flex max-w-2xl items-center justify-between">
            <h1 className="font-heading text-3xl text-gator-600">Sessions</h1>
            <div className="flex items-center gap-3">
              <button
                onClick={handleNewChat}
                className="flex items-center gap-2 rounded-xl bg-amber-accent px-5 py-2.5 font-body text-gator-600 shadow-sm transition-colors hover:bg-amber-accent/80"
              >
                <Plus size={18} />
                New Chat
              </button>
              <button
                onClick={handleLogout}
                className="rounded-lg p-2.5 text-gator-500/50 transition-colors hover:bg-gator-100 hover:text-gator-500"
                title="Sign out"
              >
                <LogOut size={18} />
              </button>
            </div>
          </div>
          {user && (
            <p className="mx-auto mt-1 max-w-2xl font-body text-xs text-gator-500/40">
              {user.email}
            </p>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          <div className="mx-auto max-w-2xl space-y-2">
            {loading && (
              <div className="flex justify-center py-12">
                <GatorSpinner size={48} />
              </div>
            )}

            {!loading && sessions.length === 0 && (
              <div className="py-16 text-center">
                <MessageSquare
                  size={48}
                  className="mx-auto mb-4 text-gator-500/30"
                  strokeWidth={1.5}
                />
                <p className="font-heading text-xl text-gator-500/50">
                  No sessions yet
                </p>
                <p className="mt-1 font-body text-sm text-gator-500/40">
                  Start a new chat to train your first model
                </p>
              </div>
            )}

            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => router.push(`/chat/${session.id}`)}
                className="flex w-full items-center gap-4 rounded-xl bg-cream/60 px-5 py-4 text-left shadow-sm transition-colors hover:bg-cream"
              >
                <MessageSquare
                  size={20}
                  className="shrink-0 text-gator-500/50"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-body text-base text-gator-600">
                    {session.title || "Untitled session"}
                  </p>
                  <p className="font-body text-xs text-gator-500/50">
                    {new Date(session.created_at).toLocaleDateString()}
                  </p>
                </div>
                <span
                  className={`rounded-full px-2.5 py-0.5 font-body text-xs ${
                    session.status === "completed"
                      ? "bg-gator-100 text-gator-500"
                      : session.status === "training"
                        ? "bg-amber-accent/20 text-amber-accent"
                        : "bg-gator-50 text-gator-500/50"
                  }`}
                >
                  {session.status}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <AuthGuard>
      <ChatListContent />
    </AuthGuard>
  );
}
