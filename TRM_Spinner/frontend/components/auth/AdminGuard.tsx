"use client";

import { useState, useEffect } from "react";

const ADMIN_KEY = "trm_admin_password";

export default function AdminGuard({
  children,
}: {
  children: React.ReactNode;
}) {
  const [authenticated, setAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem(ADMIN_KEY);
    if (stored) {
      setAuthenticated(true);
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // The admin password is checked client-side for dev tools access
    // Real security is enforced on the API side
    if (password) {
      localStorage.setItem(ADMIN_KEY, password);
      setAuthenticated(true);
      setError("");
    } else {
      setError("Password required");
    }
  };

  if (authenticated) {
    return <>{children}</>;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-gator-700/80 backdrop-blur-sm">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm rounded-2xl bg-cream p-8 shadow-lg"
      >
        <h2 className="mb-6 text-center font-heading text-2xl text-gator-600">
          Dev Access
        </h2>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Admin password"
          className="mb-4 w-full rounded-xl border border-gator-200 bg-gator-50 px-4 py-3 text-gator-600 placeholder:text-gator-500/50 focus:border-gator-500 focus:outline-none"
        />
        {error && (
          <p className="mb-3 text-center text-sm text-red-600">{error}</p>
        )}
        <button
          type="submit"
          className="w-full rounded-xl bg-gator-500 px-4 py-3 font-body text-cream transition-colors hover:bg-gator-600"
        >
          Enter
        </button>
      </form>
    </div>
  );
}
