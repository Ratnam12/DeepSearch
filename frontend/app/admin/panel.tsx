"use client";

import {
  ArrowLeftIcon,
  CheckCircleIcon,
  Loader2Icon,
  SearchIcon,
  ShieldCheckIcon,
  UserIcon,
} from "lucide-react";
import Link from "next/link";
import { type FormEvent, useCallback, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

type SerialisedCredits = {
  userId: string;
  email: string | null;
  bonusChat: number;
  bonusDeepSearch: number;
  notes: string | null;
  createdAt: string;
  updatedAt: string;
};

type FoundUser = {
  userId: string;
  email: string;
  fullName: string | null;
  imageUrl: string;
  createdAt: number | null;
  usage: {
    chat: { used: number; baseLimit: number; bonus: number };
    deepSearch: { used: number; baseLimit: number; bonus: number };
  };
  notes: string | null;
  lastUpdatedAt: string | null;
};

const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export function AdminPanel({
  adminEmail,
  initialCredits,
}: {
  adminEmail: string;
  initialCredits: SerialisedCredits[];
}) {
  const [credits, setCredits] = useState<SerialisedCredits[]>(initialCredits);
  const [emailInput, setEmailInput] = useState("");
  const [foundUser, setFoundUser] = useState<FoundUser | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [bonusChat, setBonusChat] = useState(0);
  const [bonusDeepSearch, setBonusDeepSearch] = useState(0);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSearch = useCallback(
    async (event: FormEvent) => {
      event.preventDefault();
      const email = emailInput.trim().toLowerCase();
      if (!email) {
        return;
      }

      setSearching(true);
      setSearchError(null);
      setFoundUser(null);
      try {
        const res = await fetch(
          `${base}/api/admin/users/search?email=${encodeURIComponent(email)}`
        );
        if (!res.ok) {
          throw new Error(`Search failed (${res.status})`);
        }
        const { user } = (await res.json()) as { user: FoundUser | null };
        if (!user) {
          setSearchError(
            "No Clerk user with that email — they may not have signed up yet."
          );
          return;
        }
        setFoundUser(user);
        setBonusChat(user.usage.chat.bonus);
        setBonusDeepSearch(user.usage.deepSearch.bonus);
        setNotes(user.notes ?? "");
      } catch (err) {
        setSearchError(
          err instanceof Error ? err.message : "Unknown search error"
        );
      } finally {
        setSearching(false);
      }
    },
    [emailInput]
  );

  const handleSave = useCallback(async () => {
    if (!foundUser) {
      return;
    }
    setSaving(true);
    try {
      const res = await fetch(`${base}/api/admin/credits`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: foundUser.userId,
          email: foundUser.email,
          bonusChat,
          bonusDeepSearch,
          notes: notes.trim() || null,
        }),
      });
      if (!res.ok) {
        throw new Error(`Save failed (${res.status})`);
      }
      const { credits: row } = (await res.json()) as {
        credits: SerialisedCredits & {
          createdAt: string;
          updatedAt: string;
        };
      };
      setCredits((prev) => {
        const idx = prev.findIndex((c) => c.userId === row.userId);
        const normalised: SerialisedCredits = {
          ...row,
          createdAt:
            typeof row.createdAt === "string"
              ? row.createdAt
              : new Date(row.createdAt).toISOString(),
          updatedAt:
            typeof row.updatedAt === "string"
              ? row.updatedAt
              : new Date(row.updatedAt).toISOString(),
        };
        if (idx === -1) {
          return [normalised, ...prev];
        }
        const next = [...prev];
        next[idx] = normalised;
        return next.sort((a, b) =>
          a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0
        );
      });
      setFoundUser({
        ...foundUser,
        usage: {
          ...foundUser.usage,
          chat: { ...foundUser.usage.chat, bonus: bonusChat },
          deepSearch: {
            ...foundUser.usage.deepSearch,
            bonus: bonusDeepSearch,
          },
        },
        notes: notes.trim() || null,
        lastUpdatedAt: new Date().toISOString(),
      });
      toast.success(
        `Credits updated — ${foundUser.email} now has ${
          bonusChat + foundUser.usage.chat.baseLimit
        } chat / ${
          bonusDeepSearch + foundUser.usage.deepSearch.baseLimit
        } DeepSearch.`
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }, [foundUser, bonusChat, bonusDeepSearch, notes]);

  return (
    <div className="min-h-dvh bg-background">
      <header className="sticky top-0 z-10 border-b border-border/60 bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-2">
            <Link
              className="flex size-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              href="/"
            >
              <ArrowLeftIcon className="size-4" />
            </Link>
            <div className="flex items-center gap-2">
              <ShieldCheckIcon className="size-4 text-primary" />
              <span className="font-medium text-[14px]">Admin</span>
            </div>
          </div>
          <div className="text-[12px] text-muted-foreground">{adminEmail}</div>
        </div>
      </header>

      <main className="mx-auto flex max-w-3xl flex-col gap-8 px-6 py-8">
        <section className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <h1 className="font-semibold text-[20px]">Grant extra credits</h1>
            <p className="text-[13px] text-muted-foreground">
              Look up a user by email to see their usage and bump their
              free-tier limit.
            </p>
          </div>

          <form
            className="flex flex-col gap-2 sm:flex-row sm:items-center"
            onSubmit={handleSearch}
          >
            <div className="relative flex-1">
              <SearchIcon className="-translate-y-1/2 absolute top-1/2 left-3 size-4 text-muted-foreground" />
              <Input
                aria-label="Email"
                autoComplete="email"
                className="h-10 pl-9"
                onChange={(e) => setEmailInput(e.target.value)}
                placeholder="user@example.com"
                type="email"
                value={emailInput}
              />
            </div>
            <Button
              className="h-10 gap-2"
              disabled={searching || !emailInput.trim()}
              type="submit"
            >
              {searching ? (
                <Loader2Icon className="size-4 animate-spin" />
              ) : (
                <SearchIcon className="size-4" />
              )}
              Find user
            </Button>
          </form>

          {searchError && (
            <p className="text-[12px] text-destructive">{searchError}</p>
          )}
        </section>

        {foundUser && (
          <section className="flex flex-col gap-4 rounded-2xl border border-border/60 bg-card/50 p-5">
            <div className="flex items-start gap-3">
              <div className="flex size-10 items-center justify-center rounded-full bg-muted ring-1 ring-border/60">
                <UserIcon className="size-4 text-muted-foreground" />
              </div>
              <div className="flex flex-1 flex-col gap-0.5">
                <div className="font-medium text-[14px]">
                  {foundUser.fullName ?? foundUser.email}
                </div>
                <div className="text-[12px] text-muted-foreground">
                  {foundUser.email}
                </div>
                <div className="font-mono text-[11px] text-muted-foreground/70">
                  {foundUser.userId}
                </div>
              </div>
              {foundUser.lastUpdatedAt && (
                <div className="text-right text-[11px] text-muted-foreground">
                  <div>last updated</div>
                  <div>
                    {new Date(foundUser.lastUpdatedAt).toLocaleDateString()}
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <CreditField
                bonus={bonusChat}
                hint="Effective limit = base + bonus"
                label="Chat queries"
                onChange={setBonusChat}
                used={foundUser.usage.chat.used}
                baseLimit={foundUser.usage.chat.baseLimit}
              />
              <CreditField
                bonus={bonusDeepSearch}
                hint="Effective limit = base + bonus"
                label="DeepSearch runs"
                onChange={setBonusDeepSearch}
                used={foundUser.usage.deepSearch.used}
                baseLimit={foundUser.usage.deepSearch.baseLimit}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label
                className="text-[12px] text-muted-foreground"
                htmlFor="admin-notes"
              >
                Notes (optional)
              </label>
              <Input
                className="h-9"
                id="admin-notes"
                onChange={(e) => setNotes(e.target.value)}
                placeholder="why you bumped them, or contact context"
                value={notes}
              />
            </div>

            <div className="flex justify-end">
              <Button
                className="gap-2"
                disabled={saving}
                onClick={handleSave}
                type="button"
              >
                {saving ? (
                  <Loader2Icon className="size-4 animate-spin" />
                ) : (
                  <CheckCircleIcon className="size-4" />
                )}
                Save credits
              </Button>
            </div>
          </section>
        )}

        <section className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between">
            <h2 className="font-semibold text-[15px]">Users with overrides</h2>
            <span className="text-[12px] text-muted-foreground">
              {credits.length} {credits.length === 1 ? "user" : "users"}
            </span>
          </div>

          {credits.length === 0 ? (
            <div className="rounded-xl border border-border/40 border-dashed bg-muted/20 px-4 py-8 text-center text-[13px] text-muted-foreground">
              No bonuses granted yet.
            </div>
          ) : (
            <div className="overflow-hidden rounded-xl border border-border/60">
              <table className="w-full text-[13px]">
                <thead className="bg-muted/30 text-[11px] text-muted-foreground uppercase tracking-wide">
                  <tr>
                    <th className="px-4 py-2 text-left font-medium">Email</th>
                    <th className="px-4 py-2 text-right font-medium">+Chat</th>
                    <th className="px-4 py-2 text-right font-medium">+DS</th>
                    <th className="px-4 py-2 text-right font-medium">
                      Updated
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {credits.map((row) => (
                    <tr
                      className="border-border/40 border-t cursor-pointer hover:bg-muted/30"
                      key={row.userId}
                      onClick={() => {
                        if (row.email) {
                          setEmailInput(row.email);
                        }
                      }}
                    >
                      <td className="px-4 py-2">
                        <div className="font-medium">
                          {row.email ?? <em>(unknown)</em>}
                        </div>
                        {row.notes && (
                          <div className="truncate text-[11px] text-muted-foreground">
                            {row.notes}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {row.bonusChat > 0 ? `+${row.bonusChat}` : "0"}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {row.bonusDeepSearch > 0
                          ? `+${row.bonusDeepSearch}`
                          : "0"}
                      </td>
                      <td className="px-4 py-2 text-right text-muted-foreground tabular-nums">
                        {new Date(row.updatedAt).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function CreditField({
  label,
  used,
  baseLimit,
  bonus,
  onChange,
  hint,
}: {
  label: string;
  used: number;
  baseLimit: number;
  bonus: number;
  onChange: (next: number) => void;
  hint: string;
}) {
  const effective = baseLimit + bonus;
  return (
    <div className="flex flex-col gap-1.5 rounded-xl border border-border/40 bg-background p-3">
      <div className="flex items-center justify-between">
        <span className="font-medium text-[12px]">{label}</span>
        <span className="text-[11px] text-muted-foreground tabular-nums">
          {used} / {effective} used
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-muted-foreground">+ bonus</span>
        <Input
          className="h-9 w-20 text-right tabular-nums"
          inputMode="numeric"
          min={0}
          onChange={(e) => {
            const n = Number.parseInt(e.target.value, 10);
            onChange(Number.isNaN(n) || n < 0 ? 0 : n);
          }}
          type="number"
          value={bonus}
        />
        <span className="text-[11px] text-muted-foreground">
          base {baseLimit}
        </span>
      </div>
      <p className="text-[10px] text-muted-foreground/70">{hint}</p>
    </div>
  );
}
