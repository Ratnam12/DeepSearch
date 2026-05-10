"use client";

import {
  HeartIcon,
  SparklesIcon,
  TelescopeIcon,
  TwitterIcon,
} from "lucide-react";
import { memo } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { OWNER_TWITTER_HANDLE } from "@/lib/constants";
import { cn } from "@/lib/utils";

// Variant chooses the headline + which usage row gets the highlight ring.
// "chat" fires when the user has used up FREE_CHAT_MESSAGE_LIMIT regular
// messages; "deepsearch" fires when they've burnt through their
// FREE_DEEPSEARCH_LIMIT research runs. Same dialog body either way — the
// CTA is identical, only the framing differs.
export type LimitReachedVariant = "chat" | "deepsearch";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  variant: LimitReachedVariant;
  chatUsed: number;
  chatLimit: number;
  deepSearchUsed: number;
  deepSearchLimit: number;
};

function PureLimitReachedDialog({
  open,
  onOpenChange,
  variant,
  chatUsed,
  chatLimit,
  deepSearchUsed,
  deepSearchLimit,
}: Props) {
  const twitterUrl = `https://x.com/${OWNER_TWITTER_HANDLE}`;
  const isDeepSearch = variant === "deepsearch";

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        className="overflow-hidden border-0 p-0 sm:max-w-[440px]"
        showCloseButton
      >
        <div
          aria-hidden="true"
          className="-z-10 absolute inset-0 bg-gradient-to-br from-primary/12 via-background to-background"
        />
        <div
          aria-hidden="true"
          className="-z-10 -translate-x-1/2 -translate-y-1/2 absolute top-0 left-1/2 size-72 rounded-full bg-primary/15 blur-3xl"
        />

        <div className="flex flex-col items-center gap-5 px-6 pt-8 pb-6 text-center">
          <div className="relative">
            <div
              aria-hidden="true"
              className="absolute inset-0 animate-pulse rounded-full bg-primary/20 blur-md"
            />
            <div className="relative flex size-14 items-center justify-center rounded-full bg-gradient-to-br from-primary/25 to-primary/5 ring-1 ring-primary/30">
              <HeartIcon className="size-6 fill-primary/80 text-primary" />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <DialogTitle className="font-semibold text-[18px] leading-tight">
              Hey, looks like you&apos;re loving DeepSearch
            </DialogTitle>
            <DialogDescription className="text-[13px] text-muted-foreground leading-relaxed">
              {isDeepSearch
                ? "You've used both your free DeepSearch runs — these multi-agent jobs are pricey to run, and right now I'm covering the bill personally."
                : "You've used all your free queries — and right now I'm covering every API call out of pocket."}{" "}
              If you want to keep going, DM me on Twitter and I&apos;ll bump
              your limit or hook you up with a pro plan.
            </DialogDescription>
          </div>

          <div className="grid w-full grid-cols-2 gap-2">
            <UsageStat
              highlight={!isDeepSearch}
              icon={<SparklesIcon className="size-3.5" />}
              label="Chat queries"
              limit={chatLimit}
              used={chatUsed}
            />
            <UsageStat
              highlight={isDeepSearch}
              icon={<TelescopeIcon className="size-3.5" />}
              label="DeepSearch"
              limit={deepSearchLimit}
              used={deepSearchUsed}
            />
          </div>

          <div className="flex w-full flex-col gap-2 pt-1">
            <Button
              asChild
              className="h-10 w-full gap-2 rounded-xl text-[13px] shadow-sm transition-transform hover:scale-[1.01]"
              size="lg"
            >
              <a href={twitterUrl} rel="noopener noreferrer" target="_blank">
                <TwitterIcon className="size-4" />
                DM me on Twitter — @{OWNER_TWITTER_HANDLE}
              </a>
            </Button>
            <Button
              className="h-9 w-full rounded-xl text-[12px] text-muted-foreground"
              onClick={() => onOpenChange(false)}
              size="sm"
              variant="ghost"
            >
              Maybe later
            </Button>
          </div>

          <p className="text-[11px] text-muted-foreground/70">
            Built by one engineer · thanks for trying it out
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function UsageStat({
  icon,
  label,
  used,
  limit,
  highlight,
}: {
  icon: React.ReactNode;
  label: string;
  used: number;
  limit: number;
  highlight: boolean;
}) {
  const exceeded = used >= limit;
  return (
    <div
      className={cn(
        "flex flex-col gap-1 rounded-xl border bg-card/50 px-3 py-2.5 text-left transition-colors",
        highlight ? "border-primary/40 bg-primary/5" : "border-border/40"
      )}
    >
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        {icon}
        <span>{label}</span>
      </div>
      <div
        className={cn(
          "font-medium text-[13px] tabular-nums",
          exceeded ? "text-foreground" : "text-foreground/80"
        )}
      >
        {used}
        <span className="text-muted-foreground/60"> / {limit}</span>
      </div>
    </div>
  );
}

export const LimitReachedDialog = memo(PureLimitReachedDialog);
