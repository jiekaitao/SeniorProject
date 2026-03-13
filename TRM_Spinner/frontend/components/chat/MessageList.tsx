"use client";

import { useEffect, useRef, type ReactNode } from "react";
import MessageBubble, { type Message } from "./MessageBubble";

interface MessageListProps {
  messages: Message[];
  children?: ReactNode;
}

export default function MessageList({ messages, children }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, children]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      {messages.length === 0 && !children && (
        <div className="flex h-full items-center justify-center">
          <p className="font-heading text-xl text-gator-500/50">
            Start a conversation...
          </p>
        </div>
      )}
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {children}
      <div ref={bottomRef} />
    </div>
  );
}
