"use client";

import ToolCallBox, { type ToolCall } from "./ToolCallBox";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: number;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
}

export default function MessageBubble({ message }: { message: Message }) {
  if (message.role === "system") {
    return (
      <div className="my-2 text-center">
        <span className="font-body text-sm italic text-gator-500/70">
          {message.content}
        </span>
      </div>
    );
  }

  const isUser = message.role === "user";

  return (
    <div className={`my-3 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-5 py-3 shadow-sm ${
          isUser
            ? "bg-gator-100 text-gator-600"
            : "bg-cream text-gator-600"
        }`}
      >
        {/* Tool call boxes (assistant only) */}
        {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mb-2">
            {message.toolCalls.map((tool) => (
              <ToolCallBox key={tool.id} tool={tool} />
            ))}
          </div>
        )}

        <p className="font-body text-base leading-relaxed whitespace-pre-wrap">
          {message.content}
          {message.isStreaming && (
            <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-gator-500 align-middle" />
          )}
        </p>
      </div>
    </div>
  );
}
