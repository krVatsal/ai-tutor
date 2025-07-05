// Google OAuth authentication utilities for frontend

export interface GoogleUser {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  profile_image_url: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: GoogleUser;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class GoogleAuthService {
  private accessToken: string | null = null;

  constructor() {
    // Load token from localStorage on initialization
    if (typeof window !== 'undefined') {
      this.accessToken = localStorage.getItem('access_token');
    }
  }

  async signInWithGoogle(googleToken: string): Promise<AuthResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/google`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          google_token: googleToken,
        }),
      });

      if (!response.ok) {
        throw new Error('Authentication failed');
      }

      const authData: AuthResponse = await response.json();
      
      // Store token in localStorage
      localStorage.setItem('access_token', authData.access_token);
      localStorage.setItem('user', JSON.stringify(authData.user));
      
      this.accessToken = authData.access_token;
      
      return authData;
    } catch (error) {
      console.error('Google sign-in error:', error);
      throw error;
    }
  }

  async getCurrentUser(): Promise<GoogleUser | null> {
    if (!this.accessToken) {
      return null;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${this.accessToken}`,
        },
      });

      if (!response.ok) {
        this.signOut();
        return null;
      }

      const user: GoogleUser = await response.json();
      localStorage.setItem('user', JSON.stringify(user));
      return user;
    } catch (error) {
      console.error('Get current user error:', error);
      this.signOut();
      return null;
    }
  }

  getToken(): string | null {
    return this.accessToken;
  }

  getUser(): GoogleUser | null {
    if (typeof window !== 'undefined') {
      const userStr = localStorage.getItem('user');
      return userStr ? JSON.parse(userStr) : null;
    }
    return null;
  }

  isSignedIn(): boolean {
    return this.accessToken !== null;
  }

  signOut(): void {
    this.accessToken = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
    }
  }

  async getAuthHeaders(): Promise<Record<string, string>> {
    const token = this.getToken();
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` }),
    };
  }
}

export const googleAuthService = new GoogleAuthService();
