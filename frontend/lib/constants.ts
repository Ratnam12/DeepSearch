export const isProductionEnvironment = process.env.NODE_ENV === "production";
export const isDevelopmentEnvironment = process.env.NODE_ENV === "development";
export const isTestEnvironment = Boolean(
  process.env.PLAYWRIGHT_TEST_BASE_URL ||
    process.env.PLAYWRIGHT ||
    process.env.CI_PLAYWRIGHT
);

export const suggestions = [
  "What are the latest developments in agentic AI research?",
  "Compare Vercel AI SDK and LangChain for production chatbots",
  "Summarise the case against speculative decoding",
  "Recent benchmarks for retrieval-augmented generation",
];
