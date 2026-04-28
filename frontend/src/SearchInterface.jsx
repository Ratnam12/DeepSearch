import { useState } from 'react';
import './App.css';

const API_BASE_URL =
  process.env.REACT_APP_API_URL || `http://${window.location.hostname}:8000`;

function parseSseChunk(buffer, onEvent) {
  const normalizedBuffer = buffer.replace(/\r\n/g, '\n');
  const parts = normalizedBuffer.split('\n\n');
  const remainder = parts.pop() || '';

  parts.forEach((part) => {
    const eventLine = part.split('\n').find((line) => line.startsWith('event:'));
    const dataLines = part
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart());

    if (!eventLine || dataLines.length === 0) {
      return;
    }

    const event = eventLine.slice(6).trim();
    const data = JSON.parse(dataLines.join('\n'));
    onEvent(event, data);
  });

  return remainder;
}

export default function SearchInterface() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [statuses, setStatuses] = useState([]);
  const [toolCalls, setToolCalls] = useState([]);
  const [ttftMs, setTtftMs] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  const handleEvent = (event, data) => {
    if (event === 'status') {
      setStatuses((current) => [...current, data.message]);
    }
    if (event === 'tool_call') {
      setToolCalls((current) => [...current, data.name]);
    }
    if (event === 'cached') {
      setStatuses((current) => [...current, 'Returned cached answer']);
      setAnswer(data.answer);
    }
    if (event === 'token') {
      if (data.ttft_ms !== undefined) {
        setTtftMs(data.ttft_ms);
      }
      setAnswer((current) => current + data.token);
    }
    if (event === 'error') {
      setStatuses((current) => [...current, `Error: ${data.message}`]);
      setIsSearching(false);
    }
    if (event === 'done') {
      setStatuses((current) => [...current, data.cached ? 'Done (cached)' : 'Done']);
      setIsSearching(false);
    }
  };

  const runSearch = async () => {
    const trimmed = question.trim();
    if (!trimmed || isSearching) {
      return;
    }

    setAnswer('');
    setStatuses([]);
    setToolCalls([]);
    setTtftMs(null);
    setIsSearching(true);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Search request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let done = false;

      while (!done) {
        const result = await reader.read();
        done = result.done;
        buffer += decoder.decode(result.value || new Uint8Array(), { stream: !done });
        buffer = parseSseChunk(buffer, handleEvent);
      }

      if (buffer.trim()) {
        parseSseChunk(`${buffer}\n\n`, handleEvent);
      }
    } catch (error) {
      setStatuses((current) => [...current, `Network error: ${error.message}`]);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <main className="search-shell">
      <section className="search-card">
        <h1>DeepSearch</h1>
        <p className="subtitle">Ask a research question and stream the grounded answer.</p>

        <div className="search-controls">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                runSearch();
              }
            }}
            placeholder="What do you want to research?"
          />
          <button type="button" disabled={isSearching} onClick={runSearch}>
            {isSearching ? 'Searching…' : 'Search'}
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
          <span>TTFT: {ttftMs === null ? '—' : `${ttftMs} ms`}</span>
        </div>

        <div className="answer-box">{answer || 'Streaming answer will appear here.'}</div>
      </section>
    </main>
  );
}
