"use client";

import { ChevronDownIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { DeepSearchMark } from "./deepsearch-mark";
import {
  type DeepSearchToolPart,
  DeepSearchToolStep,
} from "./deepsearch-tool-step";

// Wraps a run of consecutive agent tool calls in a single header that
// reads "Researched N sources" (or similar). Collapsed by default once
// the assistant text starts streaming so the answer is the focal point;
// expanded while research is still in flight so the user can watch the
// agent work.
//
// User can manually toggle either way; their choice persists and isn't
// overridden by subsequent re-renders. The auto-close on `collapsed`
// transition is one-shot — only fires when answer text first arrives,
// not on every render.

const HEADERS_BY_DOMINANT_TOOL: Record<string, string> = {
  web_search: "Searched the web",
  retrieve_chunks: "Searched memory",
  scrape_and_index: "Read sources",
  create_artifact: "Created artifact",
};

function summaryLabel(tools: DeepSearchToolPart[]): string {
  const counts = new Map<string, number>();
  for (const t of tools) {
    const name = toolName(t);
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }

  const total = tools.length;
  const allDone = tools.every((t) => t.state === "output-available");
  const verb = allDone ? "Researched" : "Researching";

  if (counts.size === 1) {
    const only = [...counts.keys()][0];
    const heading = HEADERS_BY_DOMINANT_TOOL[only] ?? "Used tool";
    return `${heading} · ${total} step${total === 1 ? "" : "s"}`;
  }

  return `${verb} · ${total} step${total === 1 ? "" : "s"}`;
}

function toolName(part: DeepSearchToolPart): string {
  if (part.type === "dynamic-tool") {
    return (part as { toolName?: string }).toolName ?? "tool";
  }
  return part.type.replace(/^tool-/, "");
}

export function DeepSearchToolGroup({
  tools,
  collapsed,
}: {
  tools: DeepSearchToolPart[];
  collapsed: boolean;
}) {
  // Track open-state in React so we can auto-close when `collapsed`
  // transitions from false → true (i.e., the answer text just arrived).
  // After that, the user has full control via clicks.
  const [open, setOpen] = useState(!collapsed);
  const [userTouched, setUserTouched] = useState(false);

  useEffect(() => {
    if (collapsed && !userTouched) {
      setOpen(false);
    }
    if (!collapsed && !userTouched) {
      setOpen(true);
    }
  }, [collapsed, userTouched]);

  if (tools.length === 0) {
    return null;
  }

  return (
    <details
      className="group/tools my-1 max-w-[min(100%,640px)]"
      onToggle={(event) => {
        const next = event.currentTarget.open;
        if (next !== open) {
          setUserTouched(true);
          setOpen(next);
        }
      }}
      open={open}
    >
      <summary
        className={cn(
          "flex items-center gap-2 py-1 text-[12.5px] leading-tight",
          "text-muted-foreground/70 hover:text-muted-foreground",
          "cursor-pointer transition-colors duration-150",
          "list-none [&::-webkit-details-marker]:hidden"
        )}
      >
        <DeepSearchMark
          className={cn("shrink-0", !collapsed && "animate-pulse")}
          size={14}
        />
        <span className="truncate font-medium">{summaryLabel(tools)}</span>
        <ChevronDownIcon
          aria-hidden
          className={cn(
            "ml-auto size-3 shrink-0 text-muted-foreground/40",
            "transition-transform duration-150",
            "group-open/tools:rotate-180"
          )}
        />
      </summary>
      <div className="mt-1 space-y-0.5 border-l border-border/40 pl-3">
        {tools.map((tool, index) => (
          <DeepSearchToolStep
            input={(tool as { input?: unknown }).input}
            key={
              (tool as { toolCallId?: string }).toolCallId ??
              `tool-step-${index}`
            }
            output={(tool as { output?: unknown }).output}
            state={
              ((tool as { state?: string }).state as
                | "input-streaming"
                | "input-available"
                | "approval-requested"
                | "approval-responded"
                | "output-available"
                | "output-error"
                | "output-denied") ?? "input-available"
            }
            type={tool.type}
          />
        ))}
      </div>
    </details>
  );
}
