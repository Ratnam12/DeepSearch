import { createHmac, timingSafeEqual } from "node:crypto";

const SIGNING_SECRET = process.env.BLOB_READ_WRITE_TOKEN ?? "";

function signPathname(pathname: string): string {
  return createHmac("sha256", SIGNING_SECRET).update(pathname).digest("hex");
}

export function createBlobAccessSignature(pathname: string): string {
  if (!SIGNING_SECRET) {
    throw new Error("BLOB_READ_WRITE_TOKEN is not configured");
  }
  return signPathname(pathname);
}

export function isValidBlobAccessSignature(
  pathname: string,
  signature: string | null
): boolean {
  if (!(SIGNING_SECRET && signature)) {
    return false;
  }

  const expected = Buffer.from(signPathname(pathname), "hex");
  const received = Buffer.from(signature, "hex");

  if (expected.byteLength !== received.byteLength) {
    return false;
  }

  return timingSafeEqual(expected, received);
}
