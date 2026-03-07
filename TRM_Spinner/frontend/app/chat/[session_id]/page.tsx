"use client";

import { use } from "react";
import AuthGuard from "@/components/auth/AuthGuard";
import ChatContainer from "@/components/chat/ChatContainer";

export default function ChatSessionPage({
  params,
}: {
  params: Promise<{ session_id: string }>;
}) {
  const { session_id } = use(params);

  return (
    <AuthGuard>
      <ChatContainer sessionId={session_id} />
    </AuthGuard>
  );
}
