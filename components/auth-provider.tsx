'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { googleAuthService, GoogleUser, AuthResponse } from '@/lib/google-auth';

interface AuthContextType {
  user: GoogleUser | null;
  isLoaded: boolean;
  isSignedIn: boolean;
  signIn: (googleToken: string) => Promise<void>;
  signOut: () => void;
  getToken: () => string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<GoogleUser | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Initialize auth state
    const initializeAuth = async () => {
      try {
        const currentUser = await googleAuthService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error('Auth initialization error:', error);
      } finally {
        setIsLoaded(true);
      }
    };

    initializeAuth();
  }, []);

  const signIn = useCallback(async (googleToken: string) => {
    try {
      const authData = await googleAuthService.signInWithGoogle(googleToken);
      setUser(authData.user);
    } catch (error) {
      console.error('Sign in error:', error);
      throw error;
    }
  }, []);

  const signOut = useCallback(() => {
    googleAuthService.signOut();
    setUser(null);
  }, []);

  const getToken = useCallback(() => {
    return googleAuthService.getToken();
  }, []);

  const value: AuthContextType = {
    user,
    isLoaded,
    isSignedIn: !!user,
    signIn,
    signOut,
    getToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export function useUser() {
  const { user, isLoaded, isSignedIn } = useAuth();
  return { user, isLoaded, isSignedIn };
}
