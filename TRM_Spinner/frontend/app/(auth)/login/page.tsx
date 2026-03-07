"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import GatorSpinner from "@/components/ui/GatorSpinner";
import { Mail, CheckCircle } from "lucide-react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { user, loading: authLoading, sendMagicLink } = useAuth();
  const router = useRouter();

  // If already logged in, redirect
  if (!authLoading && user) {
    router.replace("/chat");
    return null;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await sendMagicLink(email);
      setSent(true);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Unknown error";
      setError(`Could not send login link: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  if (authLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gator-50">
        <GatorSpinner size={64} />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gator-50 p-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <h1 className="font-heading text-4xl text-gator-600">TRM Spinner</h1>
          <p className="mt-2 font-body text-gator-500/70">
            Train tiny recursive models
          </p>
        </div>

        {sent ? (
          <div className="rounded-2xl bg-cream p-8 text-center shadow-sm">
            <CheckCircle
              size={48}
              className="mx-auto mb-4 text-gator-500"
              strokeWidth={1.5}
            />
            <h2 className="mb-2 font-heading text-2xl text-gator-600">
              Check your email
            </h2>
            <p className="font-body text-sm text-gator-500/70">
              We sent a login link to{" "}
              <span className="font-semibold text-gator-600">{email}</span>.
              Click it to sign in.
            </p>
            <button
              onClick={() => setSent(false)}
              className="mt-6 font-body text-sm text-gator-500/50 underline transition-colors hover:text-gator-500"
            >
              Use a different email
            </button>
          </div>
        ) : (
          <form
            onSubmit={handleSubmit}
            className="rounded-2xl bg-cream p-8 shadow-sm"
          >
            <h2 className="mb-6 font-heading text-2xl text-gator-600">
              Sign In
            </h2>

            <div className="relative">
              <Mail
                size={18}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-gator-500/40"
              />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email address"
                required
                className="w-full rounded-xl border border-gator-200 bg-gator-50 py-3 pl-11 pr-4 text-gator-600 placeholder:text-gator-500/40 focus:border-gator-500 focus:outline-none"
              />
            </div>

            {error && (
              <p className="mt-3 text-center font-body text-sm text-red-600">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="mt-6 flex w-full items-center justify-center rounded-xl bg-gator-600 px-4 py-3 font-body text-cream transition-colors hover:bg-gator-700 disabled:opacity-50"
            >
              {loading ? <GatorSpinner size={24} /> : "Send Login Link"}
            </button>

            <p className="mt-4 text-center font-body text-xs text-gator-500/50">
              No account? One will be created automatically.
            </p>
          </form>
        )}
      </div>
    </div>
  );
}
