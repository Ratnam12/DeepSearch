import { gateway } from "ai";
import { titleModel } from "./models";

// DeepSearch runs the agent + LLM on the FastAPI backend, so the frontend
// only needs the AI Gateway-based provider for the title-generation server
// action. The chatbot template's mock-test branch was removed alongside
// its e2e tests.

export function getLanguageModel(modelId: string) {
  return gateway.languageModel(modelId);
}

export function getTitleModel() {
  return gateway.languageModel(titleModel.id);
}
