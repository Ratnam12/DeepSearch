"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

type SseEventData = {
  answer?: string;
  cached?: boolean;
  message?: string;
  name?: string;
  token?: string;
  ttft_ms?: number;
};

function parseEventData(event: MessageEvent<string>): SseEventData {
  return JSON.parse(event.data) as SseEventData;
}

export default function SearchInterface() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [statuses, setStatuses] = useState<string[]>([]);
  const [toolCalls, setToolCalls] = useState<string[]>([]);
  const [ttftMs, setTtftMs] = useState<number | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const closeSearchStream = () => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  };

  useEffect(() => closeSearchStream, []);

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

  const handleStreamError = (message: string) => {
    setStatuses((current) => [...current, `Network error: ${message}`]);
    setIsSearching(false);
    closeSearchStream();
  };

  const streamUrl = (question: string): string => {
    if (!API_BASE_URL) {
      throw new Error("NEXT_PUBLIC_API_URL is not configured");
    }

    const url = new URL("/search/stream", `${API_BASE_URL}/`);
    url.searchParams.set("question", question);
    return url.toString();
  };

  const handleSourceError = (event: Event) => {
    if ("data" in event && typeof event.data === "string") {
      handleEvent("error", parseEventData(event as MessageEvent<string>));
      closeSearchStream();
      return;
    }

    handleStreamError("Search stream connection failed");
  };

  const bindStreamEvent = (source: EventSource, eventName: string) => {
    source.addEventListener(eventName, (event) => {
      try {
        handleEvent(eventName, parseEventData(event as MessageEvent<string>));
        if (eventName === "done") {
          closeSearchStream();
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Invalid event";
        handleStreamError(message);
      }
    });
  };

  const runSearch = () => {
    const trimmed = question.trim();
    if (!trimmed || isSearching) {
      return;
    }

    closeSearchStream();
    setAnswer("");
    setStatuses([]);
    setToolCalls([]);
    setTtftMs(null);
    setIsSearching(true);

    try {
      const source = new EventSource(streamUrl(trimmed));
      eventSourceRef.current = source;
      ["status", "tool_call", "cached", "token", "done"].forEach(
        (eventName) => bindStreamEvent(source, eventName),
      );
      source.addEventListener("error", handleSourceError);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      handleStreamError(message);
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
