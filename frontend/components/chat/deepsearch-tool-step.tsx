"use client";

import {
  ChevronDownIcon,
  FileTextIcon,
  GlobeIcon,
  type LucideIcon,
  SearchIcon,
  WrenchIcon,
} from "lucide-react";
import type { JSX } from "react";
import { cn } from "@/lib/utils";
import { DeepSearchMark } from "./deepsearch-mark";

// A deliberately understated render of a single agent tool call. Replaces
// the ai-elements <Tool> card (with its border, badge, and wrench icon)
// with an inline details/summary that reads as text — meant to feel like
// a status line the agent left behind, not a UI widget.
//
// The label is task-shaped, not function-shaped: instead of
// `retrieve_chunks` we say "Searching memory · query". The raw I/O is
// still available behind the chevron for anyone who wants to inspect it.

type ToolStepState =
  | "input-streaming"
  | "input-available"
  | "approval-requested"
  | "approval-responded"
  | "output-available"
  | "output-error"
  | "output-denied";

// Loose shape covering both static `tool-<name>` parts and
// `dynamic-tool` parts coming back from the AI SDK stream.
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

function formatPayload(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

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
  const inputText = formatPayload(input);
  const outputText = formatPayload(output);
  const hasDetails = inputText !== null || outputText !== null;

  return (
    <details className="group/tool my-0.5 max-w-[min(100%,640px)]">
      <summary
        className={cn(
          "flex items-center gap-2 py-1 text-[12.5px] leading-tight",
          "text-muted-foreground/65 hover:text-muted-foreground/90",
          "transition-colors duration-150",
          "list-none [&::-webkit-details-marker]:hidden",
          hasDetails && "cursor-pointer"
        )}
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
            <span className="animate-[pulse_1.4s_ease-in-out_0.2s_infinite]">
              ·
            </span>
            <span className="animate-[pulse_1.4s_ease-in-out_0.4s_infinite]">
              ·
            </span>
          </span>
        )}
        {hasDetails && (
          <ChevronDownIcon
            aria-hidden
            className={cn(
              "size-3 shrink-0 ml-auto",
              "opacity-0 transition-all duration-150",
              "group-hover/tool:opacity-40 group-open/tool:opacity-60",
              "group-open/tool:rotate-180"
            )}
          />
        )}
      </summary>
      {hasDetails && (
        <div className="ml-[22px] mt-1 mb-2 space-y-1.5">
          {inputText !== null && (
            <ToolPayload body={inputText} heading="Input" />
          )}
          {outputText !== null && (
            <ToolPayload body={outputText} heading="Output" />
          )}
        </div>
      )}
    </details>
  );
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
