import { notFound } from "next/navigation";
import { getAdminContext } from "@/lib/admin";
import { listUserCredits } from "@/lib/db/queries";
import { AdminPanel } from "./panel";

// 404 (not 401/403) to non-admins so the route doesn't reveal it exists.
// The admin gate runs server-side and prevents any data from leaking
// into the rendered HTML — non-admins see Next's standard not-found page.
export default async function AdminPage() {
  const ctx = await getAdminContext();
  if (!ctx) {
    notFound();
  }

  const initialCredits = await listUserCredits();

  return (
    <AdminPanel
      adminEmail={ctx.email}
      initialCredits={initialCredits.map((row) => ({
        ...row,
        createdAt: row.createdAt.toISOString(),
        updatedAt: row.updatedAt.toISOString(),
      }))}
    />
  );
}
