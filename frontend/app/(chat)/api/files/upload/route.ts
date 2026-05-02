import { auth } from "@clerk/nextjs/server";
import { put } from "@vercel/blob";
import { NextResponse } from "next/server";
import { z } from "zod";
import { createBlobAccessSignature } from "../blob-access";

// 25 MB covers iPhone HEICs and most research-paper PDFs without
// pushing past Vercel Blob's per-request body limit on Hobby (≈ 50 MB).
const MAX_BYTES = 25 * 1024 * 1024;

const ALLOWED_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  "image/heic",
  "image/heif",
  "application/pdf",
]);
const PDF_SIGNATURE = [0x25, 0x50, 0x44, 0x46];
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47];
const JPEG_SIGNATURE = [0xff, 0xd8, 0xff];
const GIF_SIGNATURE = [0x47, 0x49, 0x46];
const RIFF_SIGNATURE = [0x52, 0x49, 0x46, 0x46];

type PrivatePutOptions = {
  access: "private";
  addRandomSuffix: true;
  contentType: string;
};

type PrivatePut = (
  pathname: string,
  body: ArrayBuffer,
  options: PrivatePutOptions
) => ReturnType<typeof put>;

const putPrivate = put as unknown as PrivatePut;

const FileSchema = z.object({
  file: z
    .instanceof(File)
    .refine((file) => file.size <= MAX_BYTES, {
      message: "File size should be 25MB or less",
    }),
});

function startsWithBytes(bytes: Uint8Array, signature: number[]): boolean {
  return signature.every((byte, index) => bytes.at(index) === byte);
}

function inferContentType(file: File, buffer: ArrayBuffer): string | null {
  const declaredType = file.type.toLowerCase();
  if (ALLOWED_TYPES.has(declaredType)) {
    return declaredType;
  }

  const bytes = new Uint8Array(buffer.slice(0, 12));
  const name = file.name.toLowerCase();

  if (name.endsWith(".pdf") && startsWithBytes(bytes, PDF_SIGNATURE)) {
    return "application/pdf";
  }
  if (startsWithBytes(bytes, PNG_SIGNATURE)) {
    return "image/png";
  }
  if (startsWithBytes(bytes, JPEG_SIGNATURE)) {
    return "image/jpeg";
  }
  if (startsWithBytes(bytes, GIF_SIGNATURE)) {
    return "image/gif";
  }
  if (startsWithBytes(bytes, RIFF_SIGNATURE) && name.endsWith(".webp")) {
    return "image/webp";
  }
  if (name.endsWith(".heic") || name.endsWith(".heif")) {
    return "image/heic";
  }

  return null;
}

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
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    const validatedFile = FileSchema.safeParse({ file });

    if (!validatedFile.success) {
      const errorMessage = validatedFile.error.errors
        .map((error) => error.message)
        .join(", ");

      return NextResponse.json({ error: errorMessage }, { status: 400 });
    }

    const filename = file.name || "attachment";
    const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_") || "attachment";
    const fileBuffer = await file.arrayBuffer();
    const contentType = inferContentType(file, fileBuffer);

    if (!contentType) {
      return NextResponse.json(
        {
          error: `Unsupported file type. Allowed: ${Array.from(ALLOWED_TYPES).join(", ")}`,
        },
        { status: 400 }
      );
    }

    try {
      const data = await putPrivate(
        `uploads/${userId}/${safeName}`,
        fileBuffer,
        {
          access: "private",
          addRandomSuffix: true,
          contentType,
        }
      );

      let pageCount: number | undefined;
      if (contentType === "application/pdf") {
        try {
          const { PDFDocument } = await import("pdf-lib");
          const doc = await PDFDocument.load(fileBuffer, {
            ignoreEncryption: true,
          });
          pageCount = doc.getPageCount();
        } catch (err) {
          // Best-effort metadata — a corrupt or password-protected PDF
          // still uploads successfully; the model handles it downstream.
          console.warn("pdf-lib could not parse uploaded PDF", { err });
        }
      }

      const pathname = data.pathname;
      const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
      const previewUrl = `${basePath}/api/files/blob?pathname=${encodeURIComponent(pathname)}`;
      const modelUrl = new URL(previewUrl, request.url);
      modelUrl.searchParams.set(
        "signature",
        createBlobAccessSignature(pathname)
      );

      return NextResponse.json({
        ...data,
        blobUrl: data.url,
        contentType,
        filename,
        modelUrl: modelUrl.toString(),
        pageCount,
        url: previewUrl,
      });
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
