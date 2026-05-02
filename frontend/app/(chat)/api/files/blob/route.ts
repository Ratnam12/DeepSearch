import { auth } from "@clerk/nextjs/server";
import { get } from "@vercel/blob";
import { type NextRequest, NextResponse } from "next/server";
import { isValidBlobAccessSignature } from "../blob-access";

export async function GET(request: NextRequest) {
  const pathname = request.nextUrl.searchParams.get("pathname");

  if (!pathname) {
    return NextResponse.json({ error: "Missing pathname" }, { status: 400 });
  }

  const { userId } = await auth();
  const signature = request.nextUrl.searchParams.get("signature");

  if (!(userId || isValidBlobAccessSignature(pathname, signature))) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const result = await get(pathname, { access: "private" });

  if (!result) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  if (result.statusCode === 304) {
    return new NextResponse(null, { status: 304 });
  }

  return new NextResponse(result.stream, {
    headers: {
      "Cache-Control": "private, no-store",
      "Content-Type": result.blob.contentType || "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
