import { TutorApp } from "@/components/tutor-app";
import { auth } from '@clerk/nextjs/server';
import { redirect } from 'next/navigation';

export default async function Home() {
  const { userId } = auth();
  
  // This should not be needed due to middleware, but adding as extra security
  if (!userId) {
    redirect('/sign-up');
  }

  return (
    <main className="min-h-screen bg-background">
      <TutorApp />
    </main>
  );
}