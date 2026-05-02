"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDownIcon } from "lucide-react";
import { useEffect, useId, useState } from "react";
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
  const [open, setOpen] = useState(!collapsed);
  const [userTouched, setUserTouched] = useState(false);
  const contentId = useId();

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
    <div className="my-1 max-w-[min(100%,640px)]">
      <button
        aria-controls={contentId}
        aria-expanded={open}
        className={cn(
          "flex w-full items-center gap-2 py-1 text-[12.5px] leading-tight",
          "text-muted-foreground/70 hover:text-muted-foreground",
          "cursor-pointer transition-colors duration-150",
          "active:scale-[0.98] transition-all"
        )}
        onClick={() => {
          setUserTouched(true);
          setOpen((prev) => !prev);
        }}
        type="button"
      >
        <DeepSearchMark
          className={cn("shrink-0", !collapsed && "animate-pulse")}
          size={14}
        />
        <span className="truncate font-medium">{summaryLabel(tools)}</span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          aria-hidden
          className="ml-auto shrink-0"
          transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
        >
          <ChevronDownIcon className="size-3 text-muted-foreground/40" />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            animate={{ height: "auto", opacity: 1, y: 0 }}
            exit={{ height: 0, opacity: 0, y: -4 }}
            id={contentId}
            initial={{ height: 0, opacity: 0, y: -4 }}
            style={{ overflow: "hidden" }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
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
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
