"use client";

import { useState, useCallback } from "react";
import { FileUp, CheckCircle, AlertCircle, Loader2 } from "lucide-react";

interface DataUploadProps {
  onUploadFile: (file: File) => void;
  onSkip: () => void;
  isUploading?: boolean;
}

export default function DataUpload({
  onUploadFile,
  onSkip,
  isUploading,
}: DataUploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState<"idle" | "valid" | "invalid">("idle");
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");

  const processFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      setStatus("idle");
      setError("");
      onUploadFile(file);
    },
    [onUploadFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  return (
    <div className="p-6">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`mx-auto max-w-xl rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
          dragOver
            ? "border-gator-500 bg-gator-100/50"
            : "border-gator-200 bg-cream/30"
        }`}
      >
        {isUploading ? (
          <Loader2
            size={40}
            className="mx-auto mb-4 animate-spin text-gator-500/60"
            strokeWidth={1.5}
          />
        ) : (
          <FileUp
            size={40}
            className="mx-auto mb-4 text-gator-500/60"
            strokeWidth={1.5}
          />
        )}
        <p className="mb-2 font-heading text-xl text-gator-500">
          Drop training data
        </p>
        <p className="mb-4 font-body text-sm text-gator-500/60">
          Any text format with input/output pairs
        </p>
        <label className="inline-block cursor-pointer rounded-xl bg-gator-500 px-6 py-2 font-body text-cream transition-colors hover:bg-gator-600">
          Browse files
          <input
            type="file"
            accept=".json,.csv,.txt,.tsv,.md"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading}
          />
        </label>
        {fileName && (
          <p className="mt-3 font-body text-sm text-gator-500/60">
            {fileName}
          </p>
        )}
      </div>

      <div className="mx-auto mt-3 max-w-xl text-center">
        <button
          onClick={onSkip}
          className="font-body text-sm text-gator-500/50 underline transition-colors hover:text-gator-500"
        >
          Skip &mdash; I&apos;ll describe my data instead
        </button>
      </div>

      {status === "invalid" && (
        <div className="mx-auto mt-4 flex max-w-xl items-center gap-2 text-red-600">
          <AlertCircle size={16} />
          <span className="font-body text-sm">{error}</span>
        </div>
      )}
    </div>
  );
}
