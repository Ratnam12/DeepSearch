import { put } from "@vercel/blob";
import { NextResponse } from "next/server";
import { z } from "zod";

import { auth } from "@clerk/nextjs/server";

// 25 MB covers iPhone HEICs and most research-paper PDFs without
// pushing past Vercel Blob's per-request body limit on Hobby (≈ 50 MB).
const MAX_BYTES = 25 * 1024 * 1024;

const ALLOWED_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  "image/heic",
  "image/heif",
];

const FileSchema = z.object({
  file: z
    .instanceof(Blob)
    .refine((file) => file.size <= MAX_BYTES, {
      message: "File size should be 25MB or less",
    })
    .refine((file) => ALLOWED_TYPES.includes(file.type), {
      message: `Unsupported file type. Allowed: ${ALLOWED_TYPES.join(", ")}`,
    }),
});

export async function POST(request: Request) {
  const { userId } = await auth();

  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  if (request.body === null) {
    return new Response("Request body is empty", { status: 400 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as Blob;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    const validatedFile = FileSchema.safeParse({ file });

    if (!validatedFile.success) {
      const errorMessage = validatedFile.error.errors
        .map((error) => error.message)
        .join(", ");

      return NextResponse.json({ error: errorMessage }, { status: 400 });
    }

    const filename = (formData.get("file") as File).name;
    const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
    const fileBuffer = await file.arrayBuffer();

    try {
      const data = await put(`${safeName}`, fileBuffer, {
        access: "public",
      });

      return NextResponse.json(data);
    } catch (error) {
      // Surface the real Blob error so prod misconfigurations (missing
      // BLOB_READ_WRITE_TOKEN, quota, transient outage) don't hide
      // behind a generic toast.
      const message = error instanceof Error ? error.message : String(error);
      console.error("Vercel Blob put() failed", { error });
      return NextResponse.json(
        { error: `Upload failed: ${message}` },
        { status: 500 }
      );
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("Failed to process upload request", { error });
    return NextResponse.json(
      { error: `Failed to process request: ${message}` },
      { status: 500 }
    );
  }
}
