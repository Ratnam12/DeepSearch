export const isProductionEnvironment = process.env.NODE_ENV === "production";
export const isDevelopmentEnvironment = process.env.NODE_ENV === "development";
export const isTestEnvironment = Boolean(
  process.env.PLAYWRIGHT_TEST_BASE_URL ||
    process.env.PLAYWRIGHT ||
    process.env.CI_PLAYWRIGHT
);

// Free-tier quotas. The chat is hosted personally and pays per-token, so
// signed-in users get a fixed allowance before the LimitReachedDialog
// asks them to reach out for a bump. Counted lifetime, not per-window —
// 200/hr Redis spam guard in ratelimit.ts is a separate concern.
//
// Override at deploy time with FREE_CHAT_LIMIT / FREE_DEEPSEARCH_LIMIT
// without a code change. The /api/usage endpoint reads the same env so
// the client and server agree on the number it shows in the modal.
export const FREE_CHAT_MESSAGE_LIMIT = Number.parseInt(
  process.env.FREE_CHAT_LIMIT ?? "20",
  10
);
export const FREE_DEEPSEARCH_LIMIT = Number.parseInt(
  process.env.FREE_DEEPSEARCH_LIMIT ?? "2",
  10
);

// Twitter handle used by the LimitReachedDialog's "DM me" CTA. Public
// because the modal renders in the browser. Set NEXT_PUBLIC_OWNER_TWITTER
// in .env.local / Vercel env to override.
export const OWNER_TWITTER_HANDLE =
  process.env.NEXT_PUBLIC_OWNER_TWITTER ?? "ratnamcodes";

// Empty-state suggestion chips. Each one should land on a substantive
// research task that exercises the agent's tool loop (web search →
// scrape → retrieve → synthesise into an artifact). Keep them concrete
// and current — vague prompts produce vague artifacts.
export const suggestions = [
  "Compare Vercel AI SDK and LangChain for production chatbots",
  "Latest advances in agentic AI research, with citations",
  "How does Cache Components work in Next.js 16?",
  "Summarise the case against speculative decoding",
];
