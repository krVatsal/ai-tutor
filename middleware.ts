import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

// Define public routes that don't require authentication
const isPublicRoute = createRouteMatcher([
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/api/webhooks(.*)', // For Clerk webhooks
]);

export default clerkMiddleware(async (auth, request) => {
  try {
    // Allow public routes to pass through
    if (isPublicRoute(request)) {
      return;
    }

    // For all other routes, require authentication
    await auth.protect();
  } catch (error) {
    console.error('Clerk middleware error:', error);
    
    // If there's an authentication error, redirect to sign-in
    if (error instanceof Error && error.message.includes('jwk-kid-mismatch')) {
      const signInUrl = new URL('/sign-in', request.url);
      return Response.redirect(signInUrl);
    }
    
    // For other errors, allow the request to continue (will show error boundary)
    return;
  }
});

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
  ],
};