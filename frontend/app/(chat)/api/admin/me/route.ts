import { isAdminRequest } from "@/lib/admin";

// Returns whether the current session is the admin. The sidebar uses
// this to decide whether to show the "/admin" link without leaking the
// admin email into the client bundle.
export async function GET() {
  const isAdmin = await isAdminRequest();
  return Response.json({ isAdmin });
}
