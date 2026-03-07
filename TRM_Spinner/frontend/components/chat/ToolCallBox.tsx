"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Check, X } from "lucide-react";
import GatorSpinner from "@/components/ui/GatorSpinner";

export interface ToolCall {
  id: string;
  name: string;
  description: string;
  status: "running" | "done" | "error";
  result?: string;
  logs: string[];
}

const TOOL_LABELS: Record<string, string> = {
  classify_problem: "Analyzing Problem Type",
  generate_data: "Generating Training Data",
  start_training: "Starting Training",
  parse_file: "Parsing Uploaded File",
};

function StatusIcon({ status }: { status: ToolCall["status"] }) {
  if (status === "running") {
    return <GatorSpinner size={20} />;
  }
  if (status === "done") {
    return (
      <div className="flex h-5 w-5 items-center justify-center rounded-full bg-gator-500">
        <Check size={12} className="text-cream" />
      </div>
    );
  }
  return (
    <div className="flex h-5 w-5 items-center justify-center rounded-full bg-red-500">
      <X size={12} className="text-white" />
    </div>
  );
}

export default function ToolCallBox({ tool }: { tool: ToolCall }) {
  const [expanded, setExpanded] = useState(false);
  const label = TOOL_LABELS[tool.name] || tool.description || tool.name;
  const hasLogs = tool.logs.length > 0;

  return (
    <div className="my-2 rounded-2xl border border-gator-200/60 bg-gator-50 overflow-hidden">
      <button
        onClick={() => hasLogs && setExpanded(!expanded)}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left"
      >
        <StatusIcon status={tool.status} />
        <span className="flex-1 font-body text-sm text-gator-600">
          {label}
        </span>
        {tool.result && tool.status !== "running" && (
          <span className="font-body text-xs text-gator-500/70">
            {tool.result}
          </span>
        )}
        {hasLogs && (
          expanded
            ? <ChevronDown size={14} className="text-gator-500/50" />
            : <ChevronRight size={14} className="text-gator-500/50" />
        )}
      </button>
      {expanded && hasLogs && (
        <div className="border-t border-gator-200/40 bg-gator-700 px-4 py-3">
          {tool.logs.map((line, i) => (
            <p key={i} className="font-mono text-xs leading-relaxed text-gator-100">
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
