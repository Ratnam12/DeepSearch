import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// Routes that don't need a Clerk session.
// /api/chat (POST), /api/history, /api/document, /api/files, /api/suggestions
// are still protected — they are owner-only mutations or listings.
// /api/messages and /api/vote enforce their own per-row access checks.
const isPublicRoute = createRouteMatcher([
  "/",
  "/chat/(.*)",
  "/sign-in(.*)",
  "/sign-up(.*)",
  "/ping",
  "/api/messages(.*)",
  "/api/models(.*)",
  "/api/vote(.*)",
]);

export const proxy = clerkMiddleware(async (auth, request) => {
  if (!isPublicRoute(request)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    // All app routes except static assets.
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Clerk's internal asset proxy on custom domains.
    "/(_clerk|api|trpc)(.*)",
  ],
};
