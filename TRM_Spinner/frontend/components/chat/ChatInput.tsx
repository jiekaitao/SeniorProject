"use client";

import { useState, useRef } from "react";
import { Send } from "lucide-react";

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Type a message...",
}: ChatInputProps) {
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gator-200/50 bg-cream/50 p-4">
      <div className="mx-auto flex max-w-3xl items-end gap-3">
        <textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none rounded-xl border border-gator-200 bg-gator-50 px-4 py-3 font-body text-gator-600 placeholder:text-gator-500/40 focus:border-gator-500 focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || !text.trim()}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gator-500 text-cream transition-colors hover:bg-gator-600 disabled:opacity-40 disabled:hover:bg-gator-500"
          aria-label="Send message"
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}
