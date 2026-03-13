import { account } from "@/lib/appwrite";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Get JWT with a timeout so we never hang indefinitely. */
async function getAuthHeader(): Promise<Record<string, string>> {
  try {
    const jwtPromise = account.createJWT();
    const timeout = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("JWT timeout")), 3000)
    );
    const jwt = await Promise.race([jwtPromise, timeout]);
    return { Authorization: `Bearer ${jwt.jwt}` };
  } catch {
    return {};
  }
}

export async function fetchAPI<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const authHeader = await getAuthHeader();

  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...authHeader,
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }

  return res.json();
}

export async function fetchFormData<T>(
  path: string,
  formData: FormData
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const authHeader = await getAuthHeader();

  // Do NOT set Content-Type — browser sets multipart boundary automatically
  const res = await fetch(url, {
    method: "POST",
    headers: authHeader,
    body: formData,
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }

  return res.json();
}

export async function fetchBlob(path: string): Promise<Blob> {
  const url = `${API_BASE}${path}`;
  const authHeader = await getAuthHeader();

  const res = await fetch(url, { headers: authHeader });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }

  return res.blob();
}

// --- Streaming chat types and fetch ---

export type StreamEvent =
  | { type: "text_delta"; content: string }
  | { type: "tool_start"; name: string; description: string }
  | { type: "tool_progress"; name: string; message: string }
  | { type: "tool_done"; name: string; result: string }
  | { type: "tool_error"; name: string; message: string }
  | { type: "done"; state: string; classification?: string; job_id?: string }
  | { type: "error"; message: string };

export async function fetchSSE(
  path: string,
  body: Record<string, unknown>,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const url = `${API_BASE}${path}`;
  const authHeader = await getAuthHeader();

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeader,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines: "data: {...}\n\n"
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(":")) continue;
      if (trimmed.startsWith("data: ")) {
        try {
          const event = JSON.parse(trimmed.slice(6)) as StreamEvent;
          onEvent(event);
        } catch {
          // ignore non-JSON data lines
        }
      }
    }
  }

  // Process any remaining buffer
  if (buffer.trim().startsWith("data: ")) {
    try {
      const event = JSON.parse(buffer.trim().slice(6)) as StreamEvent;
      onEvent(event);
    } catch {
      // ignore
    }
  }
}

export function createSSEConnection(
  path: string,
  onMessage: (data: Record<string, unknown>) => void,
  onError?: (error: Event) => void
): EventSource {
  const url = `${API_BASE}${path}`;
  const source = new EventSource(url);

  source.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      // ignore non-JSON messages
    }
  };

  source.onerror = (event) => {
    if (onError) onError(event);
  };

  return source;
}
