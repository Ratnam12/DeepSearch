"use client";

import { ChevronRightIcon, FileTextIcon } from "lucide-react";
import { useArtifact } from "@/hooks/use-artifact";
import { cn } from "@/lib/utils";

// Inline preview card for an artifact attached to an assistant message.
// Clicking opens the artifact in the side panel via the same useArtifact
// SWR cache the live stream uses. This is the *only* way a user can
// re-open an artifact after a page reload — the live data-stream handler
// (data-stream-handler.tsx) processes the streaming protocol and never
// runs on persisted message parts. Without this card, an artifact
// generated in a previous session has no entry point in the UI.

type ArtifactPayload = {
  id: string;
  kind: "text" | "code" | "sheet";
  title: string;
  content: string;
};

const KIND_LABEL: Record<ArtifactPayload["kind"], string> = {
  text: "Document",
  code: "Code",
  sheet: "Spreadsheet",
};

export function DeepSearchArtifactCard({
  artifact,
}: {
  artifact: ArtifactPayload;
}) {
  const { setArtifact } = useArtifact();

  const open = () => {
    setArtifact({
      documentId: artifact.id,
      kind: artifact.kind,
      title: artifact.title,
      content: artifact.content,
      status: "idle",
      isVisible: true,
      boundingBox: { top: 0, left: 0, width: 0, height: 0 },
    });
  };

  return (
    <button
      className={cn(
        "group/artifact flex w-full max-w-[min(100%,640px)] items-center gap-3",
        "rounded-xl border border-border/40 bg-card/40 px-3.5 py-3 text-left",
        "transition-all duration-150 hover:-translate-y-px hover:bg-card/70 hover:shadow-[var(--shadow-card)]",
        "cursor-pointer"
      )}
      onClick={open}
      type="button"
    >
      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted/60 text-muted-foreground">
        <FileTextIcon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="truncate font-medium text-[13px] text-foreground">
          {artifact.title}
        </div>
        <div className="truncate text-[11px] text-muted-foreground/70">
          {KIND_LABEL[artifact.kind]} · click to open
        </div>
      </div>
      <ChevronRightIcon
        aria-hidden
        className="size-4 shrink-0 text-muted-foreground/40 transition-transform duration-150 group-hover/artifact:translate-x-0.5 group-hover/artifact:text-muted-foreground/70"
      />
    </button>
  );
}
