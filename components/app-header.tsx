import { BookOpen } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { UserButton, SignedIn, SignedOut, SignInButton } from '@clerk/nextjs';

export function AppHeader() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center">
        <div className="flex items-center gap-2 font-bold">
          <BookOpen className="h-6 w-6 text-primary" />
          <span className="text-xl">Mira</span>
          <span className="text-xl text-muted-foreground">AI Tutor</span>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <ThemeToggle />
          <SignedOut>
            <SignInButton mode="modal">
              <button className="text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Sign In
              </button>
            </SignInButton>
          </SignedOut>
          <SignedIn>
            <UserButton 
              appearance={{
                elements: {
                  avatarBox: "w-8 h-8",
                  userButtonPopoverCard: "bg-card border-border",
                  userButtonPopoverActionButton: "hover:bg-accent",
                }
              }}
              afterSignOutUrl="/sign-up"
            />
          </SignedIn>
        </div>
      </div>
    </header>
  );
}