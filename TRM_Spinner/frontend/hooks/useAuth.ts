"use client";

import { useState, useEffect, useCallback } from "react";
import { account, ID } from "@/lib/appwrite";
import type { Models } from "appwrite";

export function useAuth() {
  const [user, setUser] = useState<Models.User<Models.Preferences> | null>(
    null
  );
  const [loading, setLoading] = useState(true);

  const getUser = useCallback(async () => {
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
