import { SignIn } from '@clerk/nextjs';
import { ClerkErrorBoundary } from '@/components/clerk-error-boundary';

export default function SignInPage() {
  return (
    <ClerkErrorBoundary>
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary/5 via-background to-secondary/5">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-primary mb-2">Welcome Back</h1>
            <p className="text-muted-foreground">Sign in to continue learning with Mira</p>
          </div>
          <SignIn 
            appearance={{
              elements: {
                rootBox: "mx-auto",
                card: "shadow-2xl border-0 bg-card",
                headerTitle: "text-2xl font-bold text-foreground",
                headerSubtitle: "text-muted-foreground",
                socialButtonsBlockButton: "border-border hover:bg-accent",
                formButtonPrimary: "bg-primary hover:bg-primary/90 text-primary-foreground",
                footerActionLink: "text-primary hover:text-primary/80",
                identityPreviewEditButton: "text-primary hover:text-primary/80",
                formFieldInput: "border-border focus:border-primary",
                dividerLine: "bg-border",
                dividerText: "text-muted-foreground",
              }
            }}
            redirectUrl="/"
            signUpUrl="/sign-up"
          />
        </div>
      </div>
    </ClerkErrorBoundary>
  );
}