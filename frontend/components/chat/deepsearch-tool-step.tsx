"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronDownIcon,
  FileTextIcon,
  GlobeIcon,
  type LucideIcon,
  SearchIcon,
  WrenchIcon,
} from "lucide-react";
import type { JSX } from "react";
import { useState, useId } from "react";
import { cn } from "@/lib/utils";
import { DeepSearchMark } from "./deepsearch-mark";

// A deliberately understated render of a single agent tool call. The
// label is task-shaped, not function-shaped. The body is per-tool
// friendly content (source cards, memory snippets, status pills)
// rather than raw INPUT/OUTPUT JSON blobs.

type ToolStepState =
  | "input-streaming"
  | "input-available"
  | "approval-requested"
  | "approval-responded"
  | "output-available"
  | "output-error"
  | "output-denied";

export type DeepSearchToolPart = {
  type: string;
  toolCallId?: string;
  state?: ToolStepState;
  input?: unknown;
  output?: unknown;
  toolName?: string;
};

const isInProgress = (state: ToolStepState) =>
  state === "input-streaming" ||
  state === "input-available" ||
  state === "approval-requested";

const isFailure = (state: ToolStepState) =>
  state === "output-error" || state === "output-denied";

type Description = {
  label: string;
  Icon: LucideIcon | (({ className }: { className?: string }) => JSX.Element);
};

function describe(type: string, input: unknown): Description {
  const data = (input ?? {}) as Record<string, unknown>;

  switch (type) {
    case "tool-web_search": {
      const query = typeof data.query === "string" ? data.query : null;
      return {
        label: query ? `Searching the web · "${query}"` : "Searching the web",
        Icon: GlobeIcon,
      };
    }
    case "tool-retrieve_chunks": {
      const query = typeof data.query === "string" ? data.query : null;
      return {
        label: query ? `Searching memory · "${query}"` : "Searching memory",
        Icon: SearchIcon,
      };
    }
    case "tool-scrape_and_index": {
      const url = typeof data.url === "string" ? data.url : null;
      let label = "Reading source";
      if (url) {
        try {
          label = `Reading ${new URL(url).hostname.replace(/^www\./, "")}`;
        } catch {
          label = `Reading ${url}`;
        }
      }
      return { label, Icon: FileTextIcon };
    }
    case "tool-create_artifact": {
      const title = typeof data.title === "string" ? data.title : null;
      return {
        label: title ? `Drafting "${title}"` : "Drafting document",
        Icon: ({ className }: { className?: string }) => (
          <DeepSearchMark className={className} size={14} />
        ),
      };
    }
    default: {
      const friendly = type
        .replace(/^tool-/, "")
        .replace(/_/g, " ")
        .replace(/-/g, " ");
      return { label: friendly, Icon: WrenchIcon };
    }
  }
}

// ---------------------------------------------------------------------------
// Output parsers
// ---------------------------------------------------------------------------

type SearchResult = {
  index: number;
  title: string;
  url: string;
  host: string;
  snippet: string;
};

function parseSearchResults(text: string): SearchResult[] {
  const blocks = text.split(/\n\n+/);
  const results: SearchResult[] = [];
  for (const block of blocks) {
    // Pattern: [N] Title — URL\nSnippet
    const match = block.match(/^\[(\d+)\]\s+(.+?)\s+[—–-]\s+(https?:\/\/\S+)\n?([\s\S]*)?$/);
    if (!match) continue;
    const [, idxStr, title, url, snippet = ""] = match;
    let host = url;
    try {
      host = new URL(url).hostname.replace(/^www\./, "");
    } catch {
      // keep raw url as host
    }
    results.push({
      index: Number.parseInt(idxStr, 10),
      title: title.trim(),
      url: url.trim(),
      host,
      snippet: snippet.trim(),
    });
  }
  return results;
}

type ChunkResult = {
  index: number;
  score: number;
  sourceUrl: string;
  host: string;
  text: string;
};

function parseRetrievedChunks(text: string): ChunkResult[] {
  const blocks = text.split(/\n\n+/);
  const results: ChunkResult[] = [];
  for (const block of blocks) {
    // Pattern: [N] score=0.xxx src=URL\n<text>
    const match = block.match(/^\[(\d+)\]\s+score=([\d.]+)\s+src=(\S+)\n?([\s\S]*)?$/);
    if (!match) continue;
    const [, idxStr, scoreStr, sourceUrl, chunkText = ""] = match;
    let host = sourceUrl;
    try {
      host = new URL(sourceUrl).hostname.replace(/^www\./, "");
    } catch {
      // keep raw url
    }
    results.push({
      index: Number.parseInt(idxStr, 10),
      score: Number.parseFloat(scoreStr),
      sourceUrl,
      host,
      text: chunkText.trim(),
    });
  }
  return results;
}

function parseScrapeStatus(text: string): { chunks: number | null; url: string; failed: boolean } {
  const indexed = text.match(/Indexed (\d+) chunks? from (.+)\./);
  if (indexed) {
    return { chunks: Number.parseInt(indexed[1], 10), url: indexed[2], failed: false };
  }
  const failed = text.match(/No content extracted from (.+)\./);
  if (failed) {
    return { chunks: null, url: failed[1], failed: true };
  }
  return { chunks: null, url: "", failed: false };
}

function parseArtifactTitle(text: string): string | null {
  const match = text.match(/title='([^']+)'/);
  return match ? match[1] : null;
}

// ---------------------------------------------------------------------------
// Skeleton placeholder while a tool is still in-flight
// ---------------------------------------------------------------------------

function SkeletonLines({ count }: { count: number }) {
  return (
    <div className="space-y-2 py-1">
      {Array.from({ length: count }).map((_, i) => (
        <div
          className={cn(
            "h-3 animate-pulse rounded bg-muted-foreground/10",
            i === 0 ? "w-3/4" : i === count - 1 ? "w-1/2" : "w-full"
          )}
          // biome-ignore lint/suspicious/noArrayIndexKey: stable skeleton
          key={i}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-tool body renderers
// ---------------------------------------------------------------------------

function WebSearchBody({
  output,
  inProgress,
}: {
  output: string | null;
  inProgress: boolean;
}) {
  if (inProgress || !output) return <SkeletonLines count={3} />;

  const results = parseSearchResults(output);
  if (results.length === 0) {
    return (
      <p className="py-1 text-[11.5px] text-muted-foreground/60 italic">
        No results found.
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      {results.map((r) => (
        <a
          className={cn(
            "flex flex-col gap-0.5 rounded-md border border-border/30",
            "bg-muted/20 px-2.5 py-2 no-underline",
            "transition-colors duration-150 hover:bg-muted/40"
          )}
          href={r.url}
          key={r.index}
          rel="noopener noreferrer"
          target="_blank"
        >
          <div className="flex items-center gap-1.5">
            <img
              alt=""
              className="size-3.5 shrink-0 rounded-sm"
              loading="lazy"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
              referrerPolicy="no-referrer"
              src={`https://www.google.com/s2/favicons?domain=${r.host}&sz=32`}
            />
            <span className="truncate text-[11px] font-medium text-foreground/80">
              {r.title}
            </span>
          </div>
          <span className="text-[10.5px] text-muted-foreground/50">{r.host}</span>
          {r.snippet && (
            <p className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-muted-foreground/70">
              {r.snippet}
            </p>
          )}
        </a>
      ))}
    </div>
  );
}

function RetrieveChunksBody({
  output,
  inProgress,
}: {
  output: string | null;
  inProgress: boolean;
}) {
  if (inProgress || !output) return <SkeletonLines count={3} />;

  if (output.includes("Retrieval confidence warning")) {
    return (
      <p className="py-1 text-[11.5px] text-muted-foreground/60 italic">
        Confidence too low — searching for fresh sources.
      </p>
    );
  }

  const chunks = parseRetrievedChunks(output);
  if (chunks.length === 0) {
    return (
      <p className="py-1 text-[11.5px] text-muted-foreground/60 italic">
        No relevant results found.
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      {chunks.map((c) => (
        <div
          className={cn(
            "rounded-md border border-border/30 bg-muted/20 px-2.5 py-2"
          )}
          key={c.index}
        >
          <div className="mb-0.5 flex items-center gap-1.5">
            <span
              className={cn(
                "size-1.5 shrink-0 rounded-full",
                c.score > 0.7
                  ? "bg-emerald-500/70"
                  : c.score > 0.4
                    ? "bg-amber-400/70"
                    : "bg-muted-foreground/30"
              )}
            />
            <span className="text-[10.5px] font-medium text-muted-foreground/60">
              {c.host}
            </span>
          </div>
          <p className="line-clamp-2 text-[11px] leading-relaxed text-muted-foreground/70">
            {c.text}
          </p>
        </div>
      ))}
    </div>
  );
}

function ArtifactBody({ output }: { output: string | null }) {
  const title = output ? parseArtifactTitle(output) : null;
  return (
    <p className="py-1 text-[11.5px] text-muted-foreground/60">
      {title ? `Saved to artifact "${title}"` : "Artifact created."}
    </p>
  );
}

function FallbackBody({
  input,
  output,
  inProgress,
}: {
  input: string | null;
  output: string | null;
  inProgress: boolean;
}) {
  if (inProgress) return <SkeletonLines count={3} />;
  return (
    <div className="space-y-1.5">
      {input !== null && <ToolPayload body={input} heading="Input" />}
      {output !== null && <ToolPayload body={output} heading="Output" />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatPayload(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function ToolPayload({ heading, body }: { heading: string; body: string }) {
  return (
    <div className="space-y-0.5">
      <div className="text-[10px] font-medium uppercase tracking-[0.1em] text-muted-foreground/40">
        {heading}
      </div>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-md border border-border/30 bg-muted/20 p-2 font-mono text-[11px] leading-snug text-muted-foreground/75">
        {body}
      </pre>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function DeepSearchToolStep({
  type,
  state,
  input,
  output,
}: {
  type: string;
  state: ToolStepState;
  input: unknown;
  output: unknown;
}) {
  const { label, Icon } = describe(type, input);
  const inProgress = isInProgress(state);
  const failed = isFailure(state);
  const outputText = formatPayload(output);
  const inputText = formatPayload(input);
  const contentId = useId();

  // scrape_and_index gets an inline status pill — no expandable panel
  if (type === "tool-scrape_and_index") {
    const status = outputText ? parseScrapeStatus(outputText) : null;
    const pill = status
      ? status.failed
        ? "No content extracted"
        : status.chunks !== null
          ? `Indexed ${status.chunks} section${status.chunks === 1 ? "" : "s"}`
          : null
      : null;

    return (
      <div className="my-0.5 flex items-center gap-2 py-1 text-[12.5px] leading-tight text-muted-foreground/65">
        <Icon
          aria-hidden
          className={cn(
            "size-3.5 shrink-0",
            inProgress && "animate-pulse",
            failed && "text-destructive/80"
          )}
        />
        <span className="truncate">{label}</span>
        {inProgress && (
          <span
            aria-hidden
            className="ml-1 inline-flex gap-0.5 text-muted-foreground/40"
          >
            <span className="animate-[pulse_1.4s_ease-in-out_infinite]">·</span>
            <span className="animate-[pulse_1.4s_ease-in-out_0.2s_infinite]">·</span>
            <span className="animate-[pulse_1.4s_ease-in-out_0.4s_infinite]">·</span>
          </span>
        )}
        {!inProgress && pill && (
          <span className="ml-auto shrink-0 rounded-full bg-muted/50 px-1.5 py-0.5 text-[10px] text-muted-foreground/50">
            {pill}
          </span>
        )}
      </div>
    );
  }

  // create_artifact: always show body inline (no toggle needed — it's one line)
  if (type === "tool-create_artifact") {
    return (
      <div className="my-0.5 py-1 text-[12.5px] leading-tight">
        <div className="flex items-center gap-2 text-muted-foreground/65">
          <Icon
            aria-hidden
            className={cn(
              "size-3.5 shrink-0",
              inProgress && "animate-pulse",
              failed && "text-destructive/80"
            )}
          />
          <span className="truncate">{label}</span>
          {inProgress && (
            <span
              aria-hidden
              className="ml-1 inline-flex gap-0.5 text-muted-foreground/40"
            >
              <span className="animate-[pulse_1.4s_ease-in-out_infinite]">·</span>
              <span className="animate-[pulse_1.4s_ease-in-out_0.2s_infinite]">·</span>
              <span className="animate-[pulse_1.4s_ease-in-out_0.4s_infinite]">·</span>
            </span>
          )}
        </div>
        {!inProgress && outputText && (
          <div className="ml-[22px] mt-1">
            <ArtifactBody output={outputText} />
          </div>
        )}
      </div>
    );
  }

  // All other tools: expandable panel with animated body
  return (
    <ExpandableToolStep
      contentId={contentId}
      failed={failed}
      inProgress={inProgress}
      input={inputText}
      label={label}
      Icon={Icon}
      output={outputText}
      type={type}
    />
  );
}

function ExpandableToolStep({
  type,
  label,
  Icon,
  inProgress,
  failed,
  input,
  output,
  contentId,
}: {
  type: string;
  label: string;
  Icon: Description["Icon"];
  inProgress: boolean;
  failed: boolean;
  input: string | null;
  output: string | null;
  contentId: string;
}) {
  const [open, setOpen] = useState(false);

  // Determine whether this tool has a body worth showing
  const hasBody =
    type === "tool-web_search" ||
    type === "tool-retrieve_chunks" ||
    inProgress ||
    input !== null ||
    output !== null;

  return (
    <div className="my-0.5 max-w-[min(100%,640px)]">
      <button
        aria-controls={hasBody ? contentId : undefined}
        aria-expanded={hasBody ? open : undefined}
        className={cn(
          "flex w-full items-center gap-2 py-1 text-[12.5px] leading-tight",
          "text-muted-foreground/65 hover:text-muted-foreground/90",
          "transition-colors duration-150",
          hasBody && "cursor-pointer",
          !hasBody && "cursor-default"
        )}
        disabled={!hasBody}
        onClick={() => {
          if (hasBody) setOpen((prev) => !prev);
        }}
        type="button"
      >
        <Icon
          aria-hidden
          className={cn(
            "size-3.5 shrink-0",
            inProgress && "animate-pulse",
            failed && "text-destructive/80"
          )}
        />
        <span className="truncate">{label}</span>
        {inProgress && (
          <span
            aria-hidden
            className="ml-1 inline-flex gap-0.5 text-muted-foreground/40"
          >
            <span className="animate-[pulse_1.4s_ease-in-out_infinite]">·</span>
            <span className="animate-[pulse_1.4s_ease-in-out_0.2s_infinite]">·</span>
            <span className="animate-[pulse_1.4s_ease-in-out_0.4s_infinite]">·</span>
          </span>
        )}
        {hasBody && (
          <motion.span
            animate={{ rotate: open ? 180 : 0 }}
            aria-hidden
            className="ml-auto shrink-0"
            transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
          >
            <ChevronDownIcon
              className={cn(
                "size-3 opacity-0 transition-opacity duration-150",
                "group-hover:opacity-40",
                open && "opacity-60"
              )}
            />
          </motion.span>
        )}
      </button>

      {hasBody && (
        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              animate={{ height: "auto", opacity: 1, y: 0 }}
              exit={{ height: 0, opacity: 0, y: -4 }}
              id={contentId}
              initial={{ height: 0, opacity: 0, y: -4 }}
              style={{ overflow: "hidden" }}
              transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="ml-[22px] mt-1 mb-2">
                {type === "tool-web_search" && (
                  <WebSearchBody inProgress={inProgress} output={output} />
                )}
                {type === "tool-retrieve_chunks" && (
                  <RetrieveChunksBody inProgress={inProgress} output={output} />
                )}
                {type !== "tool-web_search" && type !== "tool-retrieve_chunks" && (
                  <FallbackBody
                    inProgress={inProgress}
                    input={input}
                    output={output}
                  />
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      )}
    </div>
  );
}
