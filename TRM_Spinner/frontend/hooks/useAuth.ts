"use client";

import { useState, useEffect, useCallback } from "react";
import { account, ID } from "@/lib/appwrite";
import type { Models } from "appwrite";

const DEV_API_KEY = process.env.NEXT_PUBLIC_DEV_API_KEY || "";
// Matches the user_id the worker's X-API-Key bypass stamps onto requests.
const DEV_USER_ID = "test-user-001";

function makeDevUser(): Models.User<Models.Preferences> {
  const now = new Date().toISOString();
  return {
    $id: DEV_USER_ID,
    $createdAt: now,
    $updatedAt: now,
    name: "Dev User",
    email: "dev@local",
    emailVerification: true,
    phone: "",
    phoneVerification: false,
    mfa: false,
    prefs: {} as Models.Preferences,
    registration: now,
    status: true,
    labels: [],
    passwordUpdate: now,
    accessedAt: now,
    targets: [],
  } as unknown as Models.User<Models.Preferences>;
}

export function useAuth() {
  const [user, setUser] = useState<Models.User<Models.Preferences> | null>(
    DEV_API_KEY ? makeDevUser() : null
  );
  const [loading, setLoading] = useState(!DEV_API_KEY);

  const getUser = useCallback(async () => {
    if (DEV_API_KEY) {
      setUser(makeDevUser());
      setLoading(false);
      return;
    }
    try {
      // Timeout so we don't hang if Appwrite is unreachable
      const timeout = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("timeout")), 5000)
      );
      const session = await Promise.race([account.get(), timeout]);
      setUser(session);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    getUser();
  }, [getUser]);

  const sendMagicLink = async (email: string) => {
    const url = `${window.location.origin}/verify`;
    await account.createMagicURLToken(ID.unique(), email, url);
  };

  const verifyMagicLink = async (userId: string, secret: string) => {
    await account.createSession(userId, secret);
    await getUser();
  };

  const logout = async () => {
    try {
      await account.deleteSession("current");
    } catch {
      // Session might already be gone
    }
    setUser(null);
  };

  return { user, loading, sendMagicLink, verifyMagicLink, logout, getUser };
}
