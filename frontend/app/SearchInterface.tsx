"use client";

import { useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

type SseEventData = {
  answer?: string;
  cached?: boolean;
  message?: string;
  name?: string;
  token?: string;
  ttft_ms?: number;
};

function parseSseChunk(
  buffer: string,
  onEvent: (event: string, data: SseEventData) => void,
): string {
  const normalizedBuffer = buffer.replace(/\r\n/g, "\n");
  const parts = normalizedBuffer.split("\n\n");
  const remainder = parts.pop() || "";

  parts.forEach((part) => {
    const lines = part.split("\n");
    const eventLine = lines.find((line) => line.startsWith("event:"));
    const dataLines = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart());

    if (!eventLine || dataLines.length === 0) {
      return;
    }

    const event = eventLine.slice(6).trim();
    const data = JSON.parse(dataLines.join("\n")) as SseEventData;
    onEvent(event, data);
  });

  return remainder;
}

export default function SearchInterface() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [statuses, setStatuses] = useState<string[]>([]);
  const [toolCalls, setToolCalls] = useState<string[]>([]);
  const [ttftMs, setTtftMs] = useState<number | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const handleEvent = (event: string, data: SseEventData) => {
    const message = data.message;
    const name = data.name;
    const token = data.token;

    if (event === "status" && message) {
      setStatuses((current) => [...current, message]);
    }
    if (event === "tool_call" && name) {
      setToolCalls((current) => [...current, name]);
    }
    if (event === "cached" && data.answer) {
      setStatuses((current) => [...current, "Returned cached answer"]);
      setAnswer(data.answer);
    }
    if (event === "token" && token) {
      if (data.ttft_ms !== undefined) {
        setTtftMs(data.ttft_ms);
      }
      setAnswer((current) => current + token);
    }
    if (event === "error") {
      setStatuses((current) => [
        ...current,
        `Error: ${data.message || "Search failed"}`,
      ]);
      setIsSearching(false);
    }
    if (event === "done") {
      setStatuses((current) => [
        ...current,
        data.cached ? "Done (cached)" : "Done",
      ]);
      setIsSearching(false);
    }
  };

  const runSearch = async () => {
    const trimmed = question.trim();
    if (!trimmed || isSearching) {
      return;
    }

    setAnswer("");
    setStatuses([]);
    setToolCalls([]);
    setTtftMs(null);
    setIsSearching(true);

    try {
      if (!API_BASE_URL) {
        throw new Error("NEXT_PUBLIC_API_URL is not configured");
      }

      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
          "Cache-Control": "no-cache",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Search request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let done = false;

      while (!done) {
        const result = await reader.read();
        done = result.done;
        buffer += decoder.decode(result.value || new Uint8Array(), {
          stream: !done,
        });
        buffer = parseSseChunk(buffer, handleEvent);
      }

      if (buffer.trim()) {
        parseSseChunk(`${buffer}\n\n`, handleEvent);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatuses((current) => [...current, `Network error: ${message}`]);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <section className="search-card">
      <h1>DeepSearch</h1>
      <p className="subtitle">Ask a research question and stream the grounded answer.</p>

      <div className="search-controls">
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              runSearch();
            }
          }}
          placeholder="What do you want to research?"
        />
        <button type="button" disabled={isSearching} onClick={runSearch}>
          {isSearching ? "Searching..." : "Search"}
        </button>
      </div>

      <div className="status-row">
        {statuses.map((status, index) => (
          <span className="status-pill" key={`${status}-${index}`}>
            {status}
          </span>
        ))}
      </div>

      <div className="tool-row">
        {toolCalls.map((tool, index) => (
          <span className="tool-badge" key={`${tool}-${index}`}>
            {tool}
          </span>
        ))}
      </div>

      <div className="metrics-row">
        <span>TTFT: {ttftMs === null ? "-" : `${ttftMs} ms`}</span>
      </div>

      <div className="answer-box">
        {answer || "Streaming answer will appear here."}
      </div>
    </section>
  );
}
