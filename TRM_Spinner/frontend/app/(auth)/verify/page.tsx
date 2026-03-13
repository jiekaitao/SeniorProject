"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import GatorSpinner from "@/components/ui/GatorSpinner";

function VerifyContent() {
  const [error, setError] = useState("");
  const { verifyMagicLink } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const userId = searchParams.get("userId");
    const secret = searchParams.get("secret");

    if (!userId || !secret) {
      setError("Invalid or expired login link.");
      return;
    }

    verifyMagicLink(userId, secret)
      .then(() => router.replace("/chat"))
      .catch(() => setError("Login link expired or already used. Please request a new one."));
  }, [searchParams, verifyMagicLink, router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gator-50 p-4">
      <div className="text-center">
        {error ? (
          <div className="rounded-2xl bg-cream p-8 shadow-sm">
            <p className="mb-4 font-body text-gator-600">{error}</p>
            <a
              href="/login"
              className="font-body text-sm text-gator-500 underline"
            >
              Back to login
            </a>
          </div>
        ) : (
          <>
            <GatorSpinner size={64} />
            <p className="mt-4 font-body text-gator-500/70">
              Signing you in...
            </p>
          </>
        )}
      </div>
    </div>
  );
}

export default function VerifyPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-gator-50">
          <GatorSpinner size={64} />
        </div>
      }
    >
      <VerifyContent />
    </Suspense>
  );
}
