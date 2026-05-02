export const isProductionEnvironment = process.env.NODE_ENV === "production";
export const isDevelopmentEnvironment = process.env.NODE_ENV === "development";
export const isTestEnvironment = Boolean(
  process.env.PLAYWRIGHT_TEST_BASE_URL ||
    process.env.PLAYWRIGHT ||
    process.env.CI_PLAYWRIGHT
);

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
