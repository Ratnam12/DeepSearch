"use client";

import { cjk } from "@streamdown/cjk";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { useMemo } from "react";
import { Streamdown } from "streamdown";
import type { ResearchSource } from "@/lib/db/schema";

// Mirrors the chat side's plugin set so reports get the same rich
// rendering — fenced code with syntax highlighting, KaTeX math,
// CJK-aware paragraph breaking, and Mermaid diagrams. The writer
// agent is encouraged in its system prompt to emit ```mermaid blocks
// where they help; this is what renders them.
const PLUGINS = { cjk, code, math, mermaid };

// Replace bare `[N]` citation markers in the report with markdown
// links pointing directly at the source URL. The writer prompt asks
// for OpenAI-style ``[N]`` citations; this turns them into clickable
// links without requiring any change to the markdown.
//
// We deliberately skip cases where ``[N]`` is already part of a
// markdown link (followed by ``(`` or preceded by ``!``) so we don't
// mangle existing structure.
function linkifyCitations(
  markdown: string,
  citations: ResearchSource[]
): string {
  if (!citations.length) return markdown;
  const urlByNum = new Map<number, string>();
  for (const c of citations) urlByNum.set(c.citationNum, c.url);

  return markdown.replace(/(?<![!\\])\[(\d+)\](?!\()/g, (match, n: string) => {
    const num = Number.parseInt(n, 10);
    const url = urlByNum.get(num);
    if (!url) return match;
    // Escaped brackets in the link text so streamdown renders the
    // visible label as ``[N]`` instead of ``N``.
    return `[\\[${num}\\]](${url})`;
  });
}

export function ResearchReportRenderer({
  markdown,
  citations,
  className,
}: {
  markdown: string;
  citations: ResearchSource[];
  className?: string;
}) {
  const linkified = useMemo(
    () => linkifyCitations(markdown, citations),
    [markdown, citations]
  );

  return (
    <Streamdown
      className={
        className ??
        // Tailwind typography baseline + a few overrides so headings,
        // tables, and code blocks read well at the report scale.
        "prose prose-sm max-w-none dark:prose-invert prose-headings:scroll-mt-20 prose-h1:font-semibold prose-h1:text-2xl prose-h2:font-semibold prose-h2:text-xl prose-h2:mt-8 prose-h3:font-medium prose-h3:text-lg prose-table:text-sm prose-pre:bg-muted/50 prose-pre:border prose-pre:border-border/60 [&_pre]:rounded-md"
      }
      plugins={PLUGINS}
    >
      {linkified}
    </Streamdown>
  );
}
