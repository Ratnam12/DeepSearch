import "server-only";

import { auth, currentUser } from "@clerk/nextjs/server";

// Single-admin gate. Hardcoded fallback so the project Just Works
// out-of-the-box for the maintainer; override per environment with
// ADMIN_EMAIL in .env.local / Vercel env when the maintainer email
// changes. There's no role/permission system — any extension here
// (multiple admins, role hierarchies) should move identity into a
// proper authz layer rather than growing this list.
export const ADMIN_EMAIL = (
  process.env.ADMIN_EMAIL ?? "ratnamsingh1201@gmail.com"
).toLowerCase();

// Returns the admin's identity if the request is authorised, or null
// otherwise. Two round-trips: auth() for the userId, then currentUser()
// for the email — there's no cheaper way to map session → email with
// Clerk's server SDK. Cache hit on the second call within the same
// request thanks to Next's request-scoped cache.
export async function getAdminContext(): Promise<{
  userId: string;
  email: string;
} | null> {
  const { userId } = await auth();
  if (!userId) {
    return null;
  }
  const user = await currentUser();
  const email = user?.primaryEmailAddress?.emailAddress?.toLowerCase();
  if (!email || email !== ADMIN_EMAIL) {
    return null;
  }
  return { userId, email };
}

export async function isAdminRequest(): Promise<boolean> {
  return (await getAdminContext()) !== null;
}
