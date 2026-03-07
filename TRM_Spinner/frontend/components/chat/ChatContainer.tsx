"use client";

import { useState, useCallback, useEffect } from "react";
import { ArrowLeft, Download } from "lucide-react";
import { useRouter } from "next/navigation";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";
import DataUpload from "./DataUpload";
import TrainingProgress from "@/components/training/TrainingProgress";
import ResearchBanner from "@/components/ui/ResearchBanner";
import type { Message } from "./MessageBubble";
import type { ToolCall } from "./ToolCallBox";
import { fetchAPI, fetchFormData, fetchBlob, fetchSSE, type StreamEvent } from "@/lib/api";

type ChatPhase =
  | "greeting"
  | "classification"
  | "data_collection"
  | "training"
  | "completed";

interface ChatContainerProps {
  sessionId: string;
}

let msgCounter = 0;
function makeId() {
  return `msg-${Date.now()}-${++msgCounter}`;
}

let toolCounter = 0;
function makeToolId() {
  return `tool-${Date.now()}-${++toolCounter}`;
}

function updateMessageFromEvent(msg: Message, event: StreamEvent): Message {
  const updated = { ...msg };

  switch (event.type) {
    case "text_delta":
      updated.content = (updated.content || "") + event.content;
      break;

    case "tool_start": {
      const newTool: ToolCall = {
        id: makeToolId(),
        name: event.name,
        description: event.description,
        status: "running",
        logs: [],
      };
      updated.toolCalls = [...(updated.toolCalls || []), newTool];
      break;
    }

    case "tool_progress": {
      const tools = [...(updated.toolCalls || [])];
      const last = tools.findLast((t) => t.name === event.name && t.status === "running");
      if (last) {
        last.logs = [...last.logs, event.message];
      }
      updated.toolCalls = tools;
      break;
    }

    case "tool_done": {
      const tools2 = [...(updated.toolCalls || [])];
      const last2 = tools2.findLast((t) => t.name === event.name && t.status === "running");
      if (last2) {
        last2.status = "done";
        last2.result = event.result;
      }
      updated.toolCalls = tools2;
      break;
    }

    case "tool_error": {
      const tools3 = [...(updated.toolCalls || [])];
      const last3 = tools3.findLast((t) => t.name === event.name && t.status === "running");
      if (last3) {
        last3.status = "error";
        last3.logs = [...last3.logs, event.message];
      }
      updated.toolCalls = tools3;
      break;
    }

    default:
      break;
  }

  return updated;
}

export default function ChatContainer({ sessionId }: ChatContainerProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [phase, setPhase] = useState<ChatPhase>("greeting");
  const [jobId, setJobId] = useState<string | null>(null);
  const [waiting, setWaiting] = useState(false);
  const [showUpload, setShowUpload] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const router = useRouter();

  // Load existing messages + session state on mount
  useEffect(() => {
    let cancelled = false;

    async function loadSession() {
      try {
        // Fetch session state and messages in parallel
        const [sessionRes, messagesRes] = await Promise.all([
          fetchAPI<{
            id: string;
            state: ChatPhase;
            job_id?: string | null;
            classification?: string | null;
          }>(`/api/sessions/${sessionId}`),
          fetchAPI<{
            documents: Array<{
              $id: string;
              role: string;
              content: string;
              created_at: string;
            }>;
          }>(`/api/sessions/${sessionId}/messages`),
        ]);

        if (cancelled) return;

        // Set phase from session
        setPhase(sessionRes.state);
        if (sessionRes.job_id) setJobId(sessionRes.job_id);

        // Load messages sorted by created_at
        const docs = messagesRes.documents || [];
        const sorted = docs.sort(
          (a, b) =>
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );

        if (sorted.length > 0) {
          setMessages(
            sorted.map((d) => ({
              id: d.$id,
              role: d.role as Message["role"],
              content: d.content,
            }))
          );
          // If we already have data, hide the upload prompt
          if (sessionRes.state !== "data_collection") {
            setShowUpload(false);
          }
        } else {
          // New session - show welcome message
          setMessages([
            {
              id: makeId(),
              role: "assistant",
              content:
                "Hey there. I can help you train a tiny recursive model on your data. What kind of task are you working on?",
            },
          ]);
        }
      } catch {
        // Session not found or network error - show fresh
        setMessages([
          {
            id: makeId(),
            role: "assistant",
            content:
              "Hey there. I can help you train a tiny recursive model on your data. What kind of task are you working on?",
          },
        ]);
      } finally {
        if (!cancelled) setLoaded(true);
      }
    }

    loadSession();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const addMessage = useCallback((role: Message["role"], content: string) => {
    setMessages((prev) => [...prev, { id: makeId(), role, content }]);
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      addMessage("user", text);
      setWaiting(true);

      // Create streaming placeholder
      const placeholderId = makeId();
      const placeholder: Message = {
        id: placeholderId,
        role: "assistant",
        content: "",
        isStreaming: true,
        toolCalls: [],
      };
      setMessages((prev) => [...prev, placeholder]);

      try {
        await fetchSSE(
          "/api/chat/stream",
          { session_id: sessionId, message: text },
          (event) => {
            if (event.type === "done") {
              const doneEvent = event as Extract<StreamEvent, { type: "done" }>;
              if (doneEvent.state) {
                setPhase(doneEvent.state as ChatPhase);
              }
              if (doneEvent.job_id) {
                setJobId(doneEvent.job_id);
                setPhase("training");
              }
              // Mark streaming as complete
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === placeholderId ? { ...m, isStreaming: false } : m
                )
              );
            } else {
              // Update the placeholder message with the new event
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === placeholderId ? updateMessageFromEvent(m, event) : m
                )
              );
            }
          },
        );

        // Ensure streaming flag is cleared after fetchSSE resolves
        setMessages((prev) =>
          prev.map((m) =>
            m.id === placeholderId ? { ...m, isStreaming: false } : m
          )
        );
      } catch {
        // Remove empty placeholder and add error
        setMessages((prev) => prev.filter((m) => m.id !== placeholderId));
        addMessage("system", "Something went wrong. Try again.");
      } finally {
        setWaiting(false);
      }
    },
    [sessionId, addMessage]
  );

  const handleFileUpload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      addMessage("system", `Uploading ${file.name}...`);

      try {
        const formData = new FormData();
        formData.append("session_id", sessionId);
        formData.append("file", file);

        const res = await fetchFormData<{
          valid: boolean;
          num_examples: number;
          max_grid_size: number;
          vocab_size: number;
          errors?: string[];
        }>("/api/data/upload-file", formData);

        if (res.valid) {
          addMessage(
            "assistant",
            `Parsed ${res.num_examples} training examples from ${file.name}. Type "start training" when ready, or upload more data.`
          );
        } else {
          addMessage(
            "system",
            `Could not parse file: ${res.errors?.join(", ") || "unknown error"}`
          );
        }
      } catch {
        addMessage("system", "Upload failed. Try a different file.");
      } finally {
        setIsUploading(false);
      }
    },
    [sessionId, addMessage]
  );

  const handleSkipUpload = useCallback(() => {
    setShowUpload(false);
    addMessage(
      "assistant",
      'No worries \u2014 you can describe what kind of data you need and I\'ll generate it. Try "generate 10 sudoku examples".'
    );
  }, [addMessage]);

  const handleTrainingComplete = useCallback(() => {
    setPhase("completed");
    addMessage("assistant", "Training complete! Your model is ready.");
  }, [addMessage]);

  const handleDownloadWeights = useCallback(async () => {
    if (!jobId) return;
    try {
      const blob = await fetchBlob(`/api/jobs/${jobId}/download`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `trm_weights_${jobId.slice(0, 8)}.pt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      addMessage("system", "Failed to download weights.");
    }
  }, [jobId, addMessage]);

  if (!loaded) {
    return (
      <div className="flex h-screen flex-col">
        <ResearchBanner />
        <div className="flex flex-1 items-center justify-center">
          <p className="font-body text-gator-500/50">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col">
      <ResearchBanner />

      {/* Header with back button */}
      <div className="flex items-center gap-3 border-b border-gator-200/50 bg-cream/30 px-4 py-2">
        <button
          onClick={() => router.push("/chat")}
          className="rounded-lg p-1.5 text-gator-500/60 transition-colors hover:bg-gator-100 hover:text-gator-500"
        >
          <ArrowLeft size={20} />
        </button>
        <span className="font-body text-sm text-gator-500/60">
          {phase === "training"
            ? "Training..."
            : phase === "completed"
              ? "Complete"
              : "Chat"}
        </span>
      </div>

      <div className="flex flex-1 flex-col overflow-hidden">
        <MessageList messages={messages}>
          {/* Training progress rendered inline at the bottom of the message list */}
          {phase === "training" && jobId && (
            <div className="my-3 flex justify-start">
              <div className="max-w-[90%] overflow-hidden rounded-2xl bg-cream shadow-sm">
                <TrainingProgress
                  jobId={jobId}
                  onComplete={handleTrainingComplete}
                />
              </div>
            </div>
          )}

          {/* Download button inline after completion */}
          {phase === "completed" && jobId && (
            <div className="my-3 flex justify-start">
              <div className="rounded-2xl bg-cream px-5 py-4 shadow-sm">
                <p className="mb-3 font-body text-sm text-gator-600">
                  Your model is ready to download.
                </p>
                <button
                  onClick={handleDownloadWeights}
                  className="inline-flex items-center gap-2 rounded-xl bg-gator-500 px-5 py-2 font-body text-cream transition-colors hover:bg-gator-600"
                >
                  <Download size={16} />
                  Download Model Weights
                </button>
              </div>
            </div>
          )}
        </MessageList>

        {phase === "data_collection" && showUpload && (
          <DataUpload
            onUploadFile={handleFileUpload}
            onSkip={handleSkipUpload}
            isUploading={isUploading}
          />
        )}

        <ChatInput
          onSend={handleSend}
          disabled={waiting}
          placeholder={
            phase === "data_collection"
              ? "Upload data above, or type to chat..."
              : phase === "training"
                ? "Training in progress \u2014 you can still chat..."
                : phase === "completed"
                  ? "Session complete \u2014 start a new chat to train another model"
                  : "Type a message..."
          }
        />
      </div>
    </div>
  );
}
