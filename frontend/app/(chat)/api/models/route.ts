import {
  chatModels,
  getAllGatewayModels,
  getCapabilities,
  isDemo,
} from "@/lib/ai/models";

export function GET() {
  const headers = {
    "Cache-Control": "public, max-age=86400, s-maxage=86400",
  };

  const capabilities = getCapabilities();

  if (isDemo) {
    return Response.json(
      { capabilities, models: getAllGatewayModels() },
      { headers }
    );
  }

  return Response.json(
    { capabilities, models: chatModels },
    { headers }
  );
}
