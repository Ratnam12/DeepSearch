// Model catalogue surfaced in the chat dropdown. Ids match OpenRouter's
// `<provider>/<model>` convention so the FastAPI agent (which talks to
// OpenRouter via the OpenAI-compatible client) can pass them through
// verbatim. The Next.js proxy forwards `selectedChatModel` as the
// `model` field on POST /chat; the agent uses it to override its
// complexity-based router (flash for simple, pro for complex).

export const DEFAULT_CHAT_MODEL = "openai/gpt-5.5";

export const titleModel = {
  id: "openai/gpt-5.4-mini",
  name: "GPT-5.4 mini",
  provider: "openai",
  description: "Fast model for title generation",
  gatewayOrder: ["openai"],
};

export type ModelCapabilities = {
  tools: boolean;
  vision: boolean;
  reasoning: boolean;
};

export type ChatModel = {
  id: string;
  name: string;
  provider: string;
  description: string;
  gatewayOrder?: string[];
  reasoningEffort?: "none" | "minimal" | "low" | "medium" | "high";
};

// Curated list of models, ordered roughly by capability tier. The agent
// will accept any OpenRouter model id, but the dropdown shows these
// because they're the ones we've validated with the tool-calling loop.
export const chatModels: ChatModel[] = [
  {
    id: "openai/gpt-5.5",
    name: "GPT-5.5",
    provider: "openai",
    description: "OpenAI flagship — best for complex multi-hop research",
  },
  {
    id: "openai/gpt-5.4-mini",
    name: "GPT-5.4 mini",
    provider: "openai",
    description: "Fast and cheap, great for short queries",
  },
  {
    id: "google/gemini-3.1-pro",
    name: "Gemini 3.1 Pro",
    provider: "google",
    description: "Google's flagship — strong reasoning and long context",
  },
  {
    id: "google/gemini-3.0-flash",
    name: "Gemini 3.0 Flash",
    provider: "google",
    description: "Google's fast tier — good for quick lookups",
  },
  {
    id: "anthropic/claude-opus-4.7",
    name: "Claude Opus 4.7",
    provider: "anthropic",
    description: "Anthropic flagship — careful, citation-friendly",
  },
  {
    id: "anthropic/claude-sonnet-4.6",
    name: "Claude Sonnet 4.6",
    provider: "anthropic",
    description: "Balanced cost and quality from Anthropic",
  },
  {
    id: "moonshotai/kimi-k2.5",
    name: "Kimi K2.5",
    provider: "moonshotai",
    description: "Moonshot AI flagship",
  },
  {
    id: "deepseek/deepseek-v3.2",
    name: "DeepSeek V3.2",
    provider: "deepseek",
    description: "Fast and capable open-weights model",
  },
  {
    id: "xai/grok-4.1",
    name: "Grok 4.1",
    provider: "xai",
    description: "xAI's latest with web-aware tool use",
  },
];

// The capability lookup in the chatbot template hit Vercel AI Gateway's
// /v1/models endpoint to populate tool/vision/reasoning flags. We don't
// route through that gateway, so we hand-mark capabilities for the
// curated list. The agent calls the OpenRouter chat-completion API which
// always supports tools; vision flags here are advisory for the UI's
// attachment button.
const CAPABILITIES: Record<string, ModelCapabilities> = {
  "openai/gpt-5.5": { tools: true, vision: true, reasoning: false },
  "openai/gpt-5.4-mini": { tools: true, vision: true, reasoning: false },
  "google/gemini-3.1-pro": { tools: true, vision: true, reasoning: true },
  "google/gemini-3.0-flash": { tools: true, vision: true, reasoning: false },
  "anthropic/claude-opus-4.7": { tools: true, vision: true, reasoning: true },
  "anthropic/claude-sonnet-4.6": { tools: true, vision: true, reasoning: true },
  "moonshotai/kimi-k2.5": { tools: true, vision: false, reasoning: false },
  "deepseek/deepseek-v3.2": { tools: true, vision: false, reasoning: false },
  "xai/grok-4.1": { tools: true, vision: true, reasoning: false },
};

export function getCapabilities(): Record<string, ModelCapabilities> {
  return CAPABILITIES;
}

export const isDemo = process.env.IS_DEMO === "1";

export type GatewayModelWithCapabilities = ChatModel & {
  capabilities: ModelCapabilities;
};

// Dynamic model discovery is disabled — DeepSearch ships a curated list
// of OpenRouter models. Returning the curated list keeps the model
// dropdown working without round-tripping to Vercel AI Gateway.
export function getAllGatewayModels(): GatewayModelWithCapabilities[] {
  return chatModels.map((model) => ({
    ...model,
    capabilities: CAPABILITIES[model.id] ?? {
      tools: false,
      vision: false,
      reasoning: false,
    },
  }));
}

export function getActiveModels(): ChatModel[] {
  return chatModels;
}

export const allowedModelIds = new Set(chatModels.map((m) => m.id));

export const modelsByProvider = chatModels.reduce(
  (acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  },
  {} as Record<string, ChatModel[]>
);
