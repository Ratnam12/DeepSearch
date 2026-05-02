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
  text: boolean;
  image: boolean;
  file: boolean;
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

type OpenRouterArchitecture = {
  input_modalities?: string[];
  output_modalities?: string[];
};

type OpenRouterModel = {
  id: string;
  name: string;
  description?: string;
  architecture?: OpenRouterArchitecture;
  supported_parameters?: string[];
};

type OpenRouterModelsResponse = {
  data?: OpenRouterModel[];
};

export type ModelCatalogue = {
  capabilities: Record<string, ModelCapabilities>;
  models: ChatModel[];
};

const OPENROUTER_MODELS_URL =
  "https://openrouter.ai/api/v1/models?output_modalities=all";
const MODEL_CATALOGUE_REVALIDATE_SECONDS = 60 * 60 * 24;

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

// Fallback only. The /api/models route uses OpenRouter's live metadata
// whenever it is reachable.
const CAPABILITIES: Record<string, ModelCapabilities> = {
  "openai/gpt-5.5": {
    text: true,
    image: true,
    file: true,
    tools: true,
    vision: true,
    reasoning: true,
  },
  "openai/gpt-5.4-mini": {
    text: true,
    image: true,
    file: true,
    tools: true,
    vision: true,
    reasoning: true,
  },
  "google/gemini-3.1-pro": {
    text: true,
    image: true,
    file: false,
    tools: true,
    vision: true,
    reasoning: true,
  },
  "google/gemini-3.0-flash": {
    text: true,
    image: true,
    file: false,
    tools: true,
    vision: true,
    reasoning: false,
  },
  "anthropic/claude-opus-4.7": {
    text: true,
    image: true,
    file: false,
    tools: true,
    vision: true,
    reasoning: true,
  },
  "anthropic/claude-sonnet-4.6": {
    text: true,
    image: true,
    file: false,
    tools: true,
    vision: true,
    reasoning: true,
  },
  "moonshotai/kimi-k2.5": {
    text: true,
    image: true,
    file: false,
    tools: true,
    vision: true,
    reasoning: true,
  },
  "deepseek/deepseek-v3.2": {
    text: true,
    image: false,
    file: false,
    tools: true,
    vision: false,
    reasoning: true,
  },
  "xai/grok-4.1": {
    text: true,
    image: true,
    file: false,
    tools: true,
    vision: true,
    reasoning: false,
  },
};

export function getCapabilities(): Record<string, ModelCapabilities> {
  return CAPABILITIES;
}

export const isDemo = process.env.IS_DEMO === "1";

export type GatewayModelWithCapabilities = ChatModel & {
  capabilities: ModelCapabilities;
};

export function getAllGatewayModels(): GatewayModelWithCapabilities[] {
  return chatModels.map((model) => ({
    ...model,
    capabilities: CAPABILITIES[model.id] ?? {
      text: false,
      image: false,
      file: false,
      tools: false,
      vision: false,
      reasoning: false,
    },
  }));
}

export function getActiveModels(): ChatModel[] {
  return chatModels;
}

function providerFromId(id: string): string {
  return id.split("/").at(0) ?? "openrouter";
}

function supportsTextOutput(model: OpenRouterModel): boolean {
  return model.architecture?.output_modalities?.includes("text") ?? false;
}

function capabilitiesFromOpenRouterModel(
  model: OpenRouterModel
): ModelCapabilities {
  const inputModalities = new Set(model.architecture?.input_modalities ?? []);
  const supportedParameters = new Set(model.supported_parameters ?? []);
  const image = inputModalities.has("image");

  return {
    text: inputModalities.has("text"),
    image,
    file: inputModalities.has("file"),
    tools: supportedParameters.has("tools"),
    vision: image,
    reasoning:
      supportedParameters.has("reasoning") ||
      supportedParameters.has("include_reasoning"),
  };
}

function toChatModel(model: OpenRouterModel, fallback?: ChatModel): ChatModel {
  return {
    id: model.id,
    name: model.name,
    provider: providerFromId(model.id),
    description:
      model.description ?? fallback?.description ?? "OpenRouter model",
    gatewayOrder: fallback?.gatewayOrder,
    reasoningEffort: fallback?.reasoningEffort,
  };
}

async function fetchOpenRouterModels(): Promise<OpenRouterModel[]> {
  const response = await fetch(OPENROUTER_MODELS_URL, {
    next: { revalidate: MODEL_CATALOGUE_REVALIDATE_SECONDS },
  });

  if (!response.ok) {
    throw new Error(`OpenRouter models request failed: ${response.status}`);
  }

  const payload = (await response.json()) as OpenRouterModelsResponse;
  return payload.data ?? [];
}

function buildCatalogue(remoteModels: OpenRouterModel[]): ModelCatalogue {
  const remoteById = new Map(remoteModels.map((model) => [model.id, model]));
  const curatedIds = new Set(chatModels.map((model) => model.id));
  const capabilities: Record<string, ModelCapabilities> = {};

  for (const model of remoteModels) {
    capabilities[model.id] = capabilitiesFromOpenRouterModel(model);
  }

  const curatedModels = chatModels.flatMap((fallback) => {
    const remoteModel = remoteById.get(fallback.id);
    return remoteModel ? [toChatModel(remoteModel, fallback)] : [];
  });
  const discoveredModels = remoteModels
    .filter(supportsTextOutput)
    .filter((model) => !curatedIds.has(model.id))
    .map((model) => toChatModel(model));

  return {
    capabilities,
    models: [...curatedModels, ...discoveredModels],
  };
}

export async function getOpenRouterModelCatalogue(): Promise<ModelCatalogue> {
  try {
    return buildCatalogue(await fetchOpenRouterModels());
  } catch {
    return {
      capabilities: getCapabilities(),
      models: chatModels,
    };
  }
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
