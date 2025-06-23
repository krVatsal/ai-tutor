import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

// Define public routes that don't require authentication
const isPublicRoute = createRouteMatcher([
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/api/webhooks(.*)', // For Clerk webhooks
]);

export default clerkMiddleware((auth, request) => {
  // Allow public routes to pass through
  if (isPublicRoute(request)) {
    return NextResponse.next();
  }

  // For all other routes, require authentication
  const { userId } = auth();
  
  if (!userId) {
    // Redirect to sign-up page if not authenticated
    const signUpUrl = new URL('/sign-up', request.url);
    return NextResponse.redirect(signUpUrl);
  }

  return NextResponse.next();
});

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
  ],
};