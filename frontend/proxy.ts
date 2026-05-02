import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// Routes that don't need a Clerk session. Everything else (including the
// chat UI, /api/chat proxy to FastAPI, history endpoints) requires sign-in.
const isPublicRoute = createRouteMatcher([
  "/sign-in(.*)",
  "/sign-up(.*)",
  "/ping",
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
